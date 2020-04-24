import gzip
from json import loads
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from pickle import dump, load
from random import random
from math import sqrt

CONTEXT_SIZE = 2
EMBEDDING_DIM = 10
MIN_WORD_COUNT = 10
TRAIN_PROPORTION = 0.7
VALID_PROPORTION = 0.15
LEARNING_RATE = 1
MOMENTUM = 0.9
BATCH_SIZE = 200
BATCHES_FOR_LOGGING = 1000
EPOCHS = 10
LEARNING_RATE_DECAY_FACTOR = 0.1
PATIENCE = 1
SUBSAMPLE_THRESHOLD = 1e-3
UNIGRAM_DISTRIBUTION_POWER = 0.75
NUM_NEGATIVE_SAMPLES = 10
UNKNOWN_TOKEN = '???'
INNER_PRODUCT_CLAMP = 4.
IMPLEMENTED_MODELS = ['CBOW', 'SGNS']
MIN_REVIEW_LENGTH = 2 * CONTEXT_SIZE + 1
CUDA = torch.cuda.is_available()


def checkAlgorithmImplemented(algorithm, implementedModels=IMPLEMENTED_MODELS):
    if algorithm.upper() not in implementedModels:
        errorMessage = 'Unknown embedding algorithm: ' + str(algorithm) + '; supported options are:'
        for model in implementedModels:
            errorMessage += ' ' + model
        errorMessage += '.'
        raise Exception(errorMessage)
    return


def subsampleProbabilityDiscard(wordFrequency, threshold):
    if wordFrequency <= 0:
        return 0
    rawResult = 1 - sqrt(threshold / wordFrequency)
    if rawResult < 0:
        return 0
    return rawResult


def subsampleWord(wordFrequency, threshold):
    if threshold is not None:
        return random() < subsampleProbabilityDiscard(wordFrequency, threshold)
    else:
        return False


def noiseDistribution(frequencies, unigramDistributionPower):
    adjustedFrequencies = [frequency ** unigramDistributionPower for frequency in frequencies]
    normalisation = sum(adjustedFrequencies)
    return [adjustedFrequency / normalisation for adjustedFrequency in adjustedFrequencies]


def produceNegativeSamples(distribution, numNegativeSamples, batchSize):
    distributions = torch.tensor(distribution, dtype=torch.float).unsqueeze(0).expand(batchSize, len(distribution))
    return torch.multinomial(distributions, num_samples=numNegativeSamples, replacement=False)


def getData(filePath):
    file = gzip.open(filePath, mode='rb')
    rawData = []
    for line in file:
        rawData.append(loads(line))
    file.close()
    print("Number of reviews:", len(rawData))
    return rawData


def preProcess(text):
    text = text.lower()
    result = ''
    for x in text:
        if x.isalpha() or x == ' ':
            result += x
    return result


def buildWordCounts(rawData):
    wordCounts = {}
    for review in rawData:
        words = preProcess(review['reviewText']).split()
        for word in words:
            if word in wordCounts:
                wordCounts[word] += 1
            else:
                wordCounts[word] = 1
    print("Number of distinct words:", len(wordCounts))
    return wordCounts


def buildVocab(rawData, minWordCount, unknownToken, unigramDistributionPower):
    wordCounts = buildWordCounts(rawData)
    allowableVocab = []
    totalRareWords = 0
    for word in wordCounts:
        if wordCounts[word] >= minWordCount:
            allowableVocab.append(word)
        else:
            totalRareWords += 1
    wordMapping = {word: i for i, word in enumerate(allowableVocab)}
    reverseWordMapping = {i: word for word, i in wordMapping.items()}
    numWords = float(sum(wordCounts[word] for word in wordCounts))
    frequencies = [wordCounts[word] / numWords for word in allowableVocab]
    if totalRareWords > 0:
        print("Words exist with total count less than", minWordCount, "which will be replaced with", unknownToken)
        reverseWordMapping[len(allowableVocab)] = unknownToken
        frequencies.append(totalRareWords / numWords)
        wordMapping[unknownToken] = len(allowableVocab)
        for word in wordCounts:
            if wordCounts[word] < minWordCount:
                wordMapping[word] = len(allowableVocab)
        allowableVocab.append(unknownToken)
    vocabularySize = len(allowableVocab)
    distribution = noiseDistribution(frequencies, unigramDistributionPower)
    print("Vocabulary size:", vocabularySize)
    return wordMapping, reverseWordMapping, allowableVocab, vocabularySize, frequencies, distribution


def preProcessWords(words, wordMapping, contextSize, algorithm, minReviewLength=MIN_REVIEW_LENGTH):
    checkAlgorithmImplemented(algorithm)
    if len(words) < minReviewLength:
        return []
    dataPoints = []
    for i in range(contextSize, len(words) - contextSize):
        context = [wordMapping[words[i - j - 1]] for j in range(contextSize)]
        context += [wordMapping[words[i + j + 1]] for j in range(contextSize)]
        target = wordMapping[words[i]]
        if algorithm.upper() == 'CBOW':
            dataPoints.append((context, target))
        elif algorithm.upper() == 'SGNS':
            for word in context:
                dataPoints.append((word, target))
    return dataPoints


def splitData(rawData, trainProportion, validProportion):
    trainData = rawData[: int(len(rawData) * trainProportion)]
    validData = rawData[int(len(rawData) * trainProportion):int(len(rawData) * (trainProportion + validProportion))]
    testData = rawData[int(len(rawData) * (trainProportion + validProportion)):]
    return trainData, validData, testData


def buildDataLoader(rawData, wordMapping, frequencies, subSample=False, batchSize=None, shuffle=False,
                    contextSize=CONTEXT_SIZE, algorithm='CBOW', threshold=SUBSAMPLE_THRESHOLD,
                    minReviewLength=MIN_REVIEW_LENGTH):
    checkAlgorithmImplemented(algorithm)
    xs = []
    ys = []
    for review in rawData:
        dataPoints = preProcessWords(preProcess(review['reviewText']).split(), wordMapping, contextSize, algorithm,
                                     minReviewLength)
        for dataPointX, dataPointY in dataPoints:
            if subSample:
                if subsampleWord(frequencies[dataPointY], threshold):
                    continue
            xs.append(dataPointX)
            ys.append(dataPointY)
    print("Size of data:", len(xs))
    xs, ys = map(torch.tensor, (xs, ys))
    ds = TensorDataset(xs, ys)
    if batchSize is not None:
        dl = DataLoader(ds, batch_size=batchSize, shuffle=shuffle)
    else:
        dl = DataLoader(ds, shuffle=shuffle)
    return dl


def setup(filePath, batchSize=BATCH_SIZE, contextSize=CONTEXT_SIZE, minWordCount=MIN_WORD_COUNT,
          unknownToken=UNKNOWN_TOKEN, trainProportion=TRAIN_PROPORTION, validProportion=VALID_PROPORTION,
          algorithm='CBOW', threshold=SUBSAMPLE_THRESHOLD, unigramDistributionPower=UNIGRAM_DISTRIBUTION_POWER):
    checkAlgorithmImplemented(algorithm)
    data = getData(filePath)
    wordMapping, reverseWordMapping, allowableVocab, vocabSize, frequencies, distribution = buildVocab(data,
                                                                                                       minWordCount,
                                                                                                       unknownToken,
                                                                                                       unigramDistributionPower)
    trainData, validData, testData = splitData(data, trainProportion, validProportion)
    print("Train data")
    trainDl = buildDataLoader(trainData, wordMapping, frequencies, subSample=True,
                              batchSize=batchSize, shuffle=True, contextSize=contextSize, algorithm=algorithm,
                              threshold=threshold)
    print("Validation data")
    validDl = buildDataLoader(validData, wordMapping, frequencies, subSample=True,
                              batchSize=2 * batchSize, shuffle=False, contextSize=contextSize, algorithm=algorithm,
                              threshold=threshold)
    print("Test data")
    testDl = buildDataLoader(testData, wordMapping, frequencies, subSample=False,
                             batchSize=2 * batchSize, shuffle=False, contextSize=contextSize, algorithm=algorithm)
    return wordMapping, reverseWordMapping, allowableVocab, vocabSize, frequencies, distribution, trainDl, validDl, testDl


class ContinuousBagOfWords(nn.Module):
    def __init__(self, vocabSize, embeddingDim, contextSize):
        super().__init__()
        self.embeddings = nn.Embedding(vocabSize, embeddingDim)
        self.linear = nn.Linear(embeddingDim, vocabSize)
        self.contextSize = contextSize
        self.embeddingDim = embeddingDim
        self.vocabSize = vocabSize

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        out = self.linear(embeds.sum(dim=-2))
        logProbabilities = F.log_softmax(out, dim=-1)
        return logProbabilities


class SkipGramWithNegativeSampling(nn.Module):
    def __init__(self, vocabSize, embeddingDim, contextSize, numNegativeSamples, innerProductClamp):
        super().__init__()
        self.embeddings = nn.Embedding(vocabSize, embeddingDim)  # These will be the inEmbeddings used in evaluation
        self.outEmbeddings = nn.Embedding(vocabSize, embeddingDim)
        self.contextSize = contextSize
        self.embeddingDim = embeddingDim
        self.vocabSize = vocabSize
        self.numNegativeSamples = numNegativeSamples
        self.innerProductClamp = innerProductClamp

        max_weight = 1 / sqrt(embeddingDim)
        torch.nn.init.uniform_(self.embeddings.weight, -max_weight, max_weight)
        torch.nn.init.uniform_(self.outEmbeddings.weight, -max_weight, max_weight)

    def forward(self, inputs, positiveOutputs, negativeOutputs):
        inputEmbeddings = self.embeddings(inputs)
        positiveOutputEmbeddings = self.outEmbeddings(positiveOutputs)
        positiveScore = torch.clamp(torch.sum(torch.mul(inputEmbeddings, positiveOutputEmbeddings), dim=1),
                                    min=-self.innerProductClamp, max=self.innerProductClamp)
        positiveScoreLogSigmoid = -F.logsigmoid(positiveScore)
        negativeOutputEmbeddings = self.outEmbeddings(negativeOutputs)
        negativeScores = torch.clamp(torch.sum(torch.mul(inputEmbeddings.unsqueeze(1), negativeOutputEmbeddings),
                                               dim=2), min=-self.innerProductClamp, max=self.innerProductClamp)
        negativeScoresLogSigmoid = torch.sum(-F.logsigmoid(-negativeScores), dim=1)

        return positiveScoreLogSigmoid + negativeScoresLogSigmoid


def train(modelName, trainDl, validDl, vocabSize, distribution=None, epochs=EPOCHS, embeddingDim=EMBEDDING_DIM,
          contextSize=CONTEXT_SIZE, innerProductClamp=INNER_PRODUCT_CLAMP, lr=LEARNING_RATE, momentum=MOMENTUM,
          numNegativeSamples=NUM_NEGATIVE_SAMPLES, learningRateDecayFactor=LEARNING_RATE_DECAY_FACTOR,
          patience=PATIENCE, algorithm='CBOW'):
    checkAlgorithmImplemented(algorithm)
    print("Training", algorithm, "for", epochs, "epochs. Context size is", contextSize, ", embedding dimension is",
          embeddingDim, ", initial learning rate is", lr, "with a decay factor of", learningRateDecayFactor, "after",
          patience, "epochs without progress.")
    trainLosses = []
    valLosses = []
    if algorithm.upper() == 'CBOW':
        model = ContinuousBagOfWords(vocabSize, embeddingDim, contextSize)
        lossFunction = nn.NLLLoss()
    elif algorithm.upper() == 'SGNS':
        model = SkipGramWithNegativeSampling(vocabSize, embeddingDim, contextSize, numNegativeSamples,
                                             innerProductClamp)
    if CUDA:
        model.cuda()
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=True)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=learningRateDecayFactor, patience=patience,
                                  verbose=True)

    for epoch in range(epochs):
        print("Epoch:", epoch)
        now = datetime.now()

        model.train()
        totalLoss = 0
        numBatchesProcessed = 0
        for xb, yb in trainDl:
            if CUDA:
                xb = xb.to('cuda')
                yb = yb.to('cuda')
            if algorithm.upper() == 'CBOW':
                predictions = model(xb)
                loss = lossFunction(predictions, yb)
            elif algorithm.upper() == 'SGNS':
                negativeSamples = produceNegativeSamples(distribution, numNegativeSamples, len(yb))
                if CUDA:
                    negativeSamples = negativeSamples.to('cuda')
                loss = torch.mean(model(yb, xb, negativeSamples))
            loss.backward()
            totalLoss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            numBatchesProcessed += 1
            if numBatchesProcessed % BATCHES_FOR_LOGGING == 0:
                print("Processed", numBatchesProcessed, "batches out of", len(trainDl), "(training)")
        trainLoss = totalLoss / len(trainDl)
        print("Training loss:", trainLoss)
        trainLosses.append(trainLoss)

        model.eval()
        with torch.no_grad():
            validLoss = 0
            numBatchesProcessed = 0
            for xb, yb in validDl:
                if CUDA:
                    xb = xb.to('cuda')
                    yb = yb.to('cuda')
                if algorithm.upper() == 'CBOW':
                    validLoss += lossFunction(model(xb), yb).item()
                elif algorithm.upper() == 'SGNS':
                    negativeSamples = produceNegativeSamples(distribution, numNegativeSamples, len(yb))
                    if CUDA:
                        negativeSamples = negativeSamples.to('cuda')
                    loss = model(yb, xb, negativeSamples)
                    validLoss += torch.mean(loss).item()
                numBatchesProcessed += 1
                if numBatchesProcessed % BATCHES_FOR_LOGGING == 0:
                    print("Processed", numBatchesProcessed, "batches out of", len(validDl), "(validation)")
        validLoss = validLoss / len(validDl)
        valLosses.append(validLoss)
        print("Validation loss:", validLoss)

        seconds = (datetime.now() - now).total_seconds()
        print("Took:", seconds)
        scheduler.step(validLoss)

        torch.save(model.state_dict(), modelName + str(epoch) + 'intermediate' + str(embeddingDim) + algorithm +
                   str(contextSize) + '.pt')

    fig, ax = plt.subplots()
    ax.plot(range(epochs), trainLosses, label="Training")
    ax.plot(range(epochs), valLosses, label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Learning curve for model", modelName)
    ax.legend()
    plt.savefig(modelName + 'learningCurve' + str(embeddingDim) + algorithm + str(contextSize) + '.png')

    return model


def saveModelState(model, modelName, wordMapping, reverseWordMapping, vocabulary, frequencies, algorithm='CBOW'):
    checkAlgorithmImplemented(algorithm)
    torch.save(model.state_dict(), modelName + algorithm + '.pt')
    outfile = open(modelName + 'WordMapping', 'wb')
    dump(wordMapping, outfile)
    outfile.close()
    outfile = open(modelName + 'reverseWordMapping', 'wb')
    dump(reverseWordMapping, outfile)
    outfile.close()
    outfile = open(modelName + 'Vocab', 'wb')
    dump(vocabulary, outfile)
    outfile.close()
    outfile = open(modelName + 'Frequencies', 'wb')
    dump(frequencies, outfile)
    outfile.close()
    if algorithm.upper() == 'CBOW':
        modelData = {'embeddingDim': model.embeddingDim, 'contextSize': model.contextSize}
    elif algorithm.upper() == 'SGNS':
        modelData = {'embeddingDim': model.embeddingDim, 'contextSize': model.contextSize,
                     'numNegativeSamples': model.numNegativeSamples, 'innerProductClamp': model.innerProductClamp}
    outfile = open(modelName + 'ModelData', 'wb')
    dump(modelData, outfile)
    outfile.close()
    print("Saved model", modelName)
    return


def loadModelState(modelName, algorithm='CBOW', unigramDistributionPower=UNIGRAM_DISTRIBUTION_POWER):
    checkAlgorithmImplemented(algorithm)
    infile = open(modelName + 'wordMapping', 'rb')
    wordMapping = load(infile)
    infile.close()
    infile = open(modelName + 'reverseWordMapping', 'rb')
    reverseWordMapping = load(infile)
    infile.close()
    infile = open(modelName + 'Vocab', 'rb')
    vocabulary = load(infile)
    infile.close()
    infile = open(modelName + 'Frequencies', 'rb')
    frequencies = load(infile)
    infile.close()
    distribution = noiseDistribution(frequencies, unigramDistributionPower)
    infile = open(modelName + 'ModelData', 'rb')
    modelData = load(infile)
    infile.close()
    if algorithm.upper() == 'CBOW':
        model = ContinuousBagOfWords(len(vocabulary), modelData['embeddingDim'], modelData['contextSize'])
    elif algorithm.upper() == 'SGNS':
        model = SkipGramWithNegativeSampling(len(vocabulary), modelData['embeddingDim'], modelData['contextSize'],
                                             modelData['numNegativeSamples'], modelData['innerProductClamp'])
    model.load_state_dict(torch.load(modelName + algorithm + '.pt'))
    print("Loaded model", modelName)
    model.eval()
    return wordMapping, reverseWordMapping, vocabulary, frequencies, distribution, model


def topKSimilarities(model, word, wordMapping, vocabulary, K=10, unknownToken=UNKNOWN_TOKEN):
    with torch.no_grad():
        wordTensor = torch.tensor(wordMapping[word], dtype=torch.long)
        if CUDA:
            wordTensor = wordTensor.to('cuda')
        wordEmbedding = model.embeddings(wordTensor)
    return topKSimilaritiesToEmbedding(model, wordEmbedding, wordMapping, vocabulary, [word], K, unknownToken)


def topKSimilaritiesToEmbedding(model, embedding, wordMapping, vocabulary, ignoreList, K, unknownToken=UNKNOWN_TOKEN):
    allSimilarities = {}
    with torch.no_grad():
        for otherWord in vocabulary:
            if otherWord == unknownToken or otherWord in ignoreList:
                continue
            otherWordTensor = torch.tensor(wordMapping[otherWord], dtype=torch.long)
            if CUDA:
                otherWordTensor = otherWordTensor.to('cuda')
            otherEmbedding = model.embeddings(otherWordTensor)
            allSimilarities[otherWord] = nn.CosineSimilarity(dim=0)(embedding, otherEmbedding).item()
    return {k: v for k, v in sorted(allSimilarities.items(), key=lambda item: item[1], reverse=True)[:K]}


def topKSimilaritiesAnalogy(model, word1, word2, word3, wordMapping, vocabulary, K=10, unknownToken=UNKNOWN_TOKEN):
    with torch.no_grad():
        word1Tensor = torch.tensor(wordMapping[word1], dtype=torch.long)
        word2Tensor = torch.tensor(wordMapping[word2], dtype=torch.long)
        word3Tensor = torch.tensor(wordMapping[word3], dtype=torch.long)
        if CUDA:
            word1Tensor = word1Tensor.to('cuda')
            word2Tensor = word2Tensor.to('cuda')
            word3Tensor = word3Tensor.to('cuda')
        word1Embedding = model.embeddings(word1Tensor)
        word2Embedding = model.embeddings(word2Tensor)
        word3Embedding = model.embeddings(word3Tensor)
        diff = word1Embedding - word2Embedding + word3Embedding
    return topKSimilaritiesToEmbedding(model, diff, wordMapping, vocabulary, [word1, word2, word3], K, unknownToken)


def finalEvaluation(model, testDl, distribution=None, lossFunction=nn.NLLLoss(), algorithm='CBOW',
                    numNegativeSamples=NUM_NEGATIVE_SAMPLES):
    checkAlgorithmImplemented(algorithm)
    with torch.no_grad():
        testLoss = 0
        numBatchesProcessed = 0
        for xb, yb in testDl:
            if CUDA:
                xb = xb.to('cuda')
                yb = yb.to('cuda')
            if algorithm.upper() == 'CBOW':
                testLoss += lossFunction(model(xb), yb).item()
            elif algorithm.upper() == 'SGNS':
                negativeSamples = produceNegativeSamples(distribution, numNegativeSamples, len(yb))
                if CUDA:
                    negativeSamples = negativeSamples.to('cuda')
                loss = model(yb, xb, negativeSamples)
                testLoss += torch.mean(loss).item()
            numBatchesProcessed += 1
            if numBatchesProcessed % BATCHES_FOR_LOGGING == 0:
                print("Processed", numBatchesProcessed, "batches out of", len(testDl), "(testing)")
        testLoss = testLoss / len(testDl)
    return testLoss


# Example usage:

algorithmType = 'CBOW'
name = 'instrumentsLowHypers'
wordIndex, reverseWordIndex, vocab, VOCAB_SIZE, wordFrequencies, sampleDistribution, trainDataLoader, validDataLoader, testDataLoader = setup('reviews_Musical_Instruments_5.json.gz', algorithm=algorithmType)
trainedModel = train(name, trainDataLoader, validDataLoader, VOCAB_SIZE, distribution=sampleDistribution,
                     algorithm=algorithmType)
print(finalEvaluation(trainedModel, testDataLoader, distribution=sampleDistribution, algorithm=algorithmType))
saveModelState(trainedModel, name, wordIndex, reverseWordIndex, vocab, wordFrequencies, algorithm=algorithmType)
# wordIndex, reverseWordIndex, vocab, wordFrequencies, sampleDistribution, loadedModel = loadModelState(name,
#                                                                                                       algorithm=algorithmType)
# print(topKSimilarities(loadedModel, 'guitar', wordIndex, vocab))
# print(topKSimilaritiesAnalogy(loadedModel, 'buying', 'buy', 'sell', wordIndex, vocab))
