from argparse import ArgumentParser
import logging
from sys import stdout
import gzip
from json import loads
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from pickle import dump, load
from random import random
from math import sqrt

CONTEXT_SIZE = 3
EMBEDDING_DIM = 16
MIN_WORD_COUNT = 10
TRAIN_PROPORTION = 0.75
VALID_PROPORTION = 0.15
LEARNING_RATE = 1
MOMENTUM = 0.9
BATCH_SIZE = 1000
BATCHES_FOR_LOGGING = 1000
EPOCHS = 20
LEARNING_RATE_DECAY_FACTOR = 0.1
PATIENCE = 1
SUBSAMPLE_THRESHOLD = 1e-3
UNIGRAM_DISTRIBUTION_POWER = 0.75
NUM_NEGATIVE_SAMPLES = 10
UNKNOWN_TOKEN = '???'
INNER_PRODUCT_CLAMP = 4.
WORD_FOR_COMPARISON = 'apple'
WORD1_FOR_ANALOGY, WORD2_FOR_ANALOGY, WORD3_FOR_ANALOGY = 'buying', 'buy', 'sell'
NUM_WORDS_FOR_COMPARISON = 10

parser = ArgumentParser(description='Training methods for word2vec')
parser.add_argument("--contextSize", "-cs", type=int, default=CONTEXT_SIZE, help="Context size for training")
parser.add_argument("--embeddingDimension", "-ed", type=int, default=EMBEDDING_DIM, help="Internal embedding dimension")
parser.add_argument("--minWordCount", "-mwc", type=int, default=MIN_WORD_COUNT,
                    help="Minimum word count to not be mapped to unknown word")
parser.add_argument("--trainProportion", "-tp", type=float, default=TRAIN_PROPORTION,
                    help="Proportion of reviews to use in training set")
parser.add_argument("--validProportion", "-vp", type=float, default=VALID_PROPORTION,
                    help="Proportion of reviews to use in validation set")
parser.add_argument("--learningRate", "-lr", type=float, default=LEARNING_RATE, help="Initial learning rate to use")
parser.add_argument("--momentum", "-m", type=float, default=MOMENTUM, help="Momentum to use in optimiser")
parser.add_argument("--batchSize", "-bs", type=int, default=BATCH_SIZE,  help="Batch size for training")
parser.add_argument("--batchesForLogging", "-bsfl", type=int, default=BATCHES_FOR_LOGGING,
                    help="After how many batches processed should the progress be logged")
parser.add_argument("--epochs", "-e", type=int, default=EPOCHS, help="How many epochs to train for")
parser.add_argument("--learningRateDecayFactor", "-lrdf", type=float, default=LEARNING_RATE_DECAY_FACTOR,
                    help="How much to reduce the learning rate when plateauing")
parser.add_argument("--patience", "-p", type=int, default=PATIENCE,
                    help="How many epochs without progress until plateau is declared")
parser.add_argument("--subsampleThreshold", "-sst", type=float, default=SUBSAMPLE_THRESHOLD,
                    help="Threshold frequency of words to begin subsampling")
parser.add_argument("--unigramDistributionPower", "-udp", type=float, default=UNIGRAM_DISTRIBUTION_POWER,
                    help="Adjustment to unigram distribution to make in selecting negative samples")
parser.add_argument("--numNegativeSamples", "-nns", type=int, default=NUM_NEGATIVE_SAMPLES,
                    help="Number of negative samples to use")
parser.add_argument("--innerProductClamp", "-ipc", type=float, default=INNER_PRODUCT_CLAMP,
                    help="How much to clamp the internal inner products")
parser.add_argument("--algorithmType", "-at", type=str, default='CBOW', help="Which algorithm to use")
parser.add_argument("--wordForComparison", "-wfc", type=str, default=WORD_FOR_COMPARISON,
                    help="Word to compare other embeddings against")
parser.add_argument("--word1ForAnalogy", "-w1fa", type=str, default=WORD1_FOR_ANALOGY,
                    help="First word in analogy task (word 2 is to word 1 as word 3 is to what?)")
parser.add_argument("--word2ForAnalogy", "-w2fa", type=str, default=WORD2_FOR_ANALOGY,
                    help="Second word in analogy task (word 2 is to word 1 as word 3 is to what?)")
parser.add_argument("--word3ForAnalogy", "-w3fa", type=str, default=WORD3_FOR_ANALOGY,
                    help="Third word in analogy task (word 2 is to word 1 as word 3 is to what?)")
parser.add_argument("--numWordsForComparison", "-nwfc", type=int, default=NUM_WORDS_FOR_COMPARISON,
                    help="Number of words to compare against in similarity or analogy tasks")

parser.add_argument("--name", "-n", type=str, required=True, default='groceries', help="Name for the model")

args = parser.parse_args()

IMPLEMENTED_MODELS = ['CBOW', 'SGNS']
MIN_REVIEW_LENGTH = 2 * args.contextSize + 1
CUDA = torch.cuda.is_available()
FULL_NAME = args.name + "CS" + str(args.contextSize) + "ED" + str(args.embeddingDimension) + "MWC" + \
            str(args.minWordCount) + "TP" + str(args.trainProportion) + "VP" + str(args.validProportion) + "LR" + \
            str(args.learningRate) + "M" + str(args.momentum) + "BS" + str(args.batchSize) + "E" + str(args.epochs) + \
            "LRDF" + str(args.learningRateDecayFactor) + "P" + str(args.patience) + "SST" + \
            str(args.subsampleThreshold) + "UDP" + str(args.unigramDistributionPower) + "NNS" + \
            str(args.numNegativeSamples) + "IPC" + str(args.innerProductClamp) + args.algorithmType


def checkAlgorithmImplemented(algorithm, logObject, implementedModels=None):
    if implementedModels is None:
        implementedModels = IMPLEMENTED_MODELS
    if algorithm.upper() not in implementedModels:
        errorMessage = 'Unknown embedding algorithm: ' + str(algorithm) + '; supported options are:'
        for model in implementedModels:
            errorMessage += ' ' + model
        errorMessage += '.'
        writeLog(errorMessage, logObject)
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


def getData(filePath, logObject):
    file = gzip.open(filePath, mode='rb')
    rawData = []
    for line in file:
        rawData.append(loads(line))
    file.close()
    writeLog("Number of reviews: " + str(len(rawData)), logObject)
    return rawData


def preProcess(text):
    text = text.lower()
    result = ''
    for x in text:
        if x.isalpha() or x == ' ':
            result += x
    return result


def buildWordCounts(rawData, logObject):
    wordCounts = {}
    for review in rawData:
        words = preProcess(review['reviewText']).split()
        for word in words:
            if word in wordCounts:
                wordCounts[word] += 1
            else:
                wordCounts[word] = 1
    writeLog("Number of distinct words: " + str(len(wordCounts)), logObject)
    return wordCounts


def buildVocab(rawData, minWordCount, unknownToken, unigramDistributionPower, logObject):
    wordCounts = buildWordCounts(rawData, logObject)
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
        writeLog("Words exist with total count less than " + str(minWordCount) + " which will be replaced with " +
                 unknownToken, logObject)
        reverseWordMapping[len(allowableVocab)] = unknownToken
        frequencies.append(totalRareWords / numWords)
        wordMapping[unknownToken] = len(allowableVocab)
        for word in wordCounts:
            if wordCounts[word] < minWordCount:
                wordMapping[word] = len(allowableVocab)
        allowableVocab.append(unknownToken)
    vocabularySize = len(allowableVocab)
    distribution = noiseDistribution(frequencies, unigramDistributionPower)
    writeLog("Vocabulary size: " + str(vocabularySize), logObject)
    return wordMapping, reverseWordMapping, allowableVocab, vocabularySize, frequencies, distribution


def preProcessWords(words, wordMapping, contextSize, algorithm, minReviewLength, logObject):
    checkAlgorithmImplemented(algorithm, logObject)
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


def buildDataLoader(rawData, wordMapping, frequencies, contextSize, algorithm, threshold, logObject,
                    minReviewLength=MIN_REVIEW_LENGTH, subSample=False, batchSize=None, shuffle=False):
    checkAlgorithmImplemented(algorithm, logObject)
    xs = []
    ys = []
    for review in rawData:
        dataPoints = preProcessWords(preProcess(review['reviewText']).split(), wordMapping, contextSize, algorithm,
                                     minReviewLength, logObject)
        for dataPointX, dataPointY in dataPoints:
            if subSample:
                if subsampleWord(frequencies[dataPointY], threshold):
                    continue
            xs.append(dataPointX)
            ys.append(dataPointY)
    writeLog("Size of data: " + str(len(xs)), logObject)
    xs, ys = map(torch.tensor, (xs, ys))
    ds = TensorDataset(xs, ys)
    if batchSize is not None:
        dl = DataLoader(ds, batch_size=batchSize, shuffle=shuffle)
    else:
        dl = DataLoader(ds, shuffle=shuffle)
    return dl


def setup(filePath, logObject, batchSize=args.batchSize, contextSize=args.contextSize, minWordCount=args.minWordCount,
          unknownToken=UNKNOWN_TOKEN, trainProportion=args.trainProportion, validProportion=args.validProportion,
          algorithm=args.algorithmType, threshold=args.subsampleThreshold,
          unigramDistributionPower=args.unigramDistributionPower):
    checkAlgorithmImplemented(algorithm, logObject)
    now = datetime.now()
    data = getData(filePath, logObject)
    wordMapping, reverseWordMapping, allowableVocab, vocabSize, frequencies, distribution = buildVocab(data,
                                                                                                       minWordCount,
                                                                                                       unknownToken,
                                                                                                       unigramDistributionPower,
                                                                                                       logObject)
    trainData, validData, testData = splitData(data, trainProportion, validProportion)
    writeLog("Train data", logObject)
    trainDl = buildDataLoader(trainData, wordMapping, frequencies, contextSize, algorithm, threshold, logObject,
                              subSample=True, batchSize=batchSize, shuffle=True)
    writeLog("Validation data", logObject)
    validDl = buildDataLoader(validData, wordMapping, frequencies, contextSize, algorithm, threshold, logObject,
                              subSample=True, batchSize=2 * batchSize, shuffle=False)
    writeLog("Test data", logObject)
    testDl = buildDataLoader(testData, wordMapping, frequencies, contextSize, algorithm, threshold, logObject,
                             subSample=False, batchSize=2 * batchSize, shuffle=False)
    seconds = (datetime.now() - now).total_seconds()
    writeLog("Setting up took: " + str(seconds) + " seconds", logObject)
    return wordMapping, reverseWordMapping, allowableVocab, vocabSize, frequencies, distribution, trainDl, validDl, testDl


class ContinuousBagOfWords(nn.Module):
    def __init__(self, vocabSize, embeddingDim, contextSize, name):
        super().__init__()
        self.embeddings = nn.Embedding(vocabSize, embeddingDim)
        self.linear = nn.Linear(embeddingDim, vocabSize)
        self.contextSize = contextSize
        self.embeddingDim = embeddingDim
        self.vocabSize = vocabSize
        self.name = name
        self.algorithmType = 'CBOW'

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        out = self.linear(embeds.sum(dim=-2))
        logProbabilities = F.log_softmax(out, dim=-1)
        return logProbabilities


class SkipGramWithNegativeSampling(nn.Module):
    def __init__(self, vocabSize, embeddingDim, contextSize, numNegativeSamples, innerProductClamp, name):
        super().__init__()
        self.embeddings = nn.Embedding(vocabSize, embeddingDim)  # These will be the inEmbeddings used in evaluation
        self.outEmbeddings = nn.Embedding(vocabSize, embeddingDim)
        self.contextSize = contextSize
        self.embeddingDim = embeddingDim
        self.vocabSize = vocabSize
        self.numNegativeSamples = numNegativeSamples
        self.innerProductClamp = innerProductClamp
        self.name = name
        self.algorithmType = 'SGNS'

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


def train(modelName, trainDl, validDl, vocabSize, logObject, distribution=None, epochs=args.epochs,
          embeddingDim=args.embeddingDimension, contextSize=args.contextSize, innerProductClamp=args.innerProductClamp,
          lr=args.learningRate, momentum=args.momentum, numNegativeSamples=args.numNegativeSamples,
          learningRateDecayFactor=args.learningRateDecayFactor, patience=args.patience, algorithm=args.algorithmType):
    checkAlgorithmImplemented(algorithm, logObject)
    writeLog("Training " + algorithm + " for " + str(epochs) + " epochs. Initial learning rate is " + str(lr) +
             " with a decay factor of " + str(learningRateDecayFactor) + " after " + str(patience) +
             " epochs without progress.", logObject)
    trainLosses = []
    valLosses = []
    if algorithm.upper() == 'CBOW':
        model = ContinuousBagOfWords(vocabSize, embeddingDim, contextSize, modelName)
        lossFunction = nn.NLLLoss()
    elif algorithm.upper() == 'SGNS':
        model = SkipGramWithNegativeSampling(vocabSize, embeddingDim, contextSize, numNegativeSamples,
                                             innerProductClamp, modelName)
    if CUDA:
        model.cuda()
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=True)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=learningRateDecayFactor, patience=patience,
                                  verbose=True)

    for epoch in range(epochs):
        now = datetime.now()
        writeLog("Epoch: " + str(epoch), logObject)

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
            if numBatchesProcessed % args.batchesForLogging == 0:
                writeLog("Processed " + str(numBatchesProcessed) + " batches out of " + str(len(trainDl)) +
                         " (training)", logObject)
        trainLoss = totalLoss / len(trainDl)
        writeLog("Training loss: " + str(trainLoss), logObject)
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
                if numBatchesProcessed % args.batchesForLogging == 0:
                    writeLog("Processed " + str(numBatchesProcessed) + " batches out of " + str(len(validDl)) +
                             " (validation)", logObject)
        validLoss = validLoss / len(validDl)
        valLosses.append(validLoss)
        writeLog("Validation loss: " + str(validLoss), logObject)

        seconds = (datetime.now() - now).total_seconds()
        writeLog("Epoch took: " + str(seconds) + " seconds", logObject)
        scheduler.step(validLoss)

        torch.save(model.state_dict(), modelName + str(epoch) + 'intermediate' + str(embeddingDim) + algorithm +
                   str(contextSize) + '.pt')

    fig, ax = plt.subplots()
    ax.plot(range(epochs), trainLosses, label="Training")
    ax.plot(range(epochs), valLosses, label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Learning curve for model " + str(modelName))
    ax.legend()
    plt.savefig(modelName + 'learningCurve' + str(embeddingDim) + algorithm + str(contextSize) + '.png')

    return model


def saveModelState(model, wordMapping, reverseWordMapping, vocabulary, frequencies, logObject):
    torch.save(model.state_dict(), model.name + model.algorithmType + '.pt')
    outfile = open(model.name + 'WordMapping', 'wb')
    dump(wordMapping, outfile)
    outfile.close()
    outfile = open(model.name + 'reverseWordMapping', 'wb')
    dump(reverseWordMapping, outfile)
    outfile.close()
    outfile = open(model.name + 'Vocab', 'wb')
    dump(vocabulary, outfile)
    outfile.close()
    outfile = open(model.name + 'Frequencies', 'wb')
    dump(frequencies, outfile)
    outfile.close()
    if model.algorithmType == 'CBOW':
        modelData = {'embeddingDim': model.embeddingDim, 'contextSize': model.contextSize}
    elif model.algorithmType == 'SGNS':
        modelData = {'embeddingDim': model.embeddingDim, 'contextSize': model.contextSize,
                     'numNegativeSamples': model.numNegativeSamples, 'innerProductClamp': model.innerProductClamp}
    outfile = open(model.name + model.algorithmType + 'ModelData', 'wb')
    dump(modelData, outfile)
    outfile.close()
    writeLog("Saved model " + model.name, logObject)
    return


def loadModelState(modelName, logObject, algorithm=args.algorithmType,
                   unigramDistributionPower=args.unigramDistributionPower):
    checkAlgorithmImplemented(algorithm, logObject)
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
    infile = open(modelName + algorithm + 'ModelData', 'rb')
    modelData = load(infile)
    infile.close()
    if algorithm.upper() == 'CBOW':
        model = ContinuousBagOfWords(len(vocabulary), modelData['embeddingDim'], modelData['contextSize'], modelName)
    elif algorithm.upper() == 'SGNS':
        model = SkipGramWithNegativeSampling(len(vocabulary), modelData['embeddingDim'], modelData['contextSize'],
                                             modelData['numNegativeSamples'], modelData['innerProductClamp'], modelName)
    model.load_state_dict(torch.load(modelName + algorithm + '.pt'))
    if CUDA:
        model.cuda()
    writeLog("Loaded model " + modelName, logObject)
    model.eval()
    return wordMapping, reverseWordMapping, vocabulary, frequencies, distribution, model


def topKSimilarities(model, word, wordMapping, vocabulary, logObject, K=10, unknownToken=UNKNOWN_TOKEN):
    if word.lower() not in wordMapping:
        writeLog(word + " not in vocabulary", logObject)
        return {}
    with torch.no_grad():
        wordTensor = torch.tensor(wordMapping[word.lower()], dtype=torch.long)
        if CUDA:
            wordTensor = wordTensor.to('cuda')
            model.cuda()
        wordEmbedding = model.embeddings(wordTensor)
    results = topKSimilaritiesToEmbedding(model, wordEmbedding, wordMapping, vocabulary, [word.lower()], K, unknownToken)
    writeLog("Most similar words to " + word, logObject)
    for result in results:
        writeLog(result + " (score = " + str(results[result]) + ")", logObject)
    return results


def topKSimilaritiesToEmbedding(model, embedding, wordMapping, vocabulary, ignoreList, K, unknownToken=UNKNOWN_TOKEN):
    allSimilarities = {}
    with torch.no_grad():
        for otherWord in vocabulary:
            if otherWord == unknownToken or otherWord in ignoreList:
                continue
            otherWordTensor = torch.tensor(wordMapping[otherWord], dtype=torch.long)
            if CUDA:
                otherWordTensor = otherWordTensor.to('cuda')
                model.cuda()
            otherEmbedding = model.embeddings(otherWordTensor)
            allSimilarities[otherWord] = nn.CosineSimilarity(dim=0)(embedding, otherEmbedding).item()
    return {k: v for k, v in sorted(allSimilarities.items(), key=lambda item: item[1], reverse=True)[:K]}


def topKSimilaritiesAnalogy(model, word1, word2, word3, wordMapping, vocabulary, logObject, K=10,
                            unknownToken=UNKNOWN_TOKEN):
    unknownWord = False
    if word1.lower() not in wordMapping:
        writeLog(word1 + " not in vocabulary", logObject)
        unknownToken = True
    if word2.lower() not in wordMapping:
        writeLog(word2 + " not in vocabulary", logObject)
        unknownWord = True
    if word3.lower() not in wordMapping:
        writeLog(word3 + " not in vocabulary", logObject)
        unknownWord = True
    if unknownWord:
        return {}
    with torch.no_grad():
        word1Tensor = torch.tensor(wordMapping[word1.lower()], dtype=torch.long)
        word2Tensor = torch.tensor(wordMapping[word2.lower()], dtype=torch.long)
        word3Tensor = torch.tensor(wordMapping[word3.lower()], dtype=torch.long)
        if CUDA:
            word1Tensor = word1Tensor.to('cuda')
            word2Tensor = word2Tensor.to('cuda')
            word3Tensor = word3Tensor.to('cuda')
            model.cuda()
        word1Embedding = model.embeddings(word1Tensor)
        word2Embedding = model.embeddings(word2Tensor)
        word3Embedding = model.embeddings(word3Tensor)
        diff = word1Embedding - word2Embedding + word3Embedding
    results = topKSimilaritiesToEmbedding(model, diff, wordMapping, vocabulary,
                                          [word1.lower(), word2.lower(), word3.lower()], K, unknownToken)
    writeLog("Most similar words to complete the analogy " + word2 + ":" + word1 + "::" + word3 + ":___", logObject)
    for result in results:
        writeLog(result + " (score = " + str(results[result]) + ")", logObject)
    return results


def finalEvaluation(model, testDl, logObject, distribution=None, lossFunction=nn.NLLLoss(),
                    numNegativeSamples=args.numNegativeSamples):
    now = datetime.now()
    with torch.no_grad():
        loss = 0
        numBatchesProcessed = 0
        for xb, yb in testDl:
            if CUDA:
                xb = xb.to('cuda')
                yb = yb.to('cuda')
                model.cuda()
            if model.algorithmType == 'CBOW':
                loss += lossFunction(model(xb), yb).item()
            elif model.algorithmType == 'SGNS':
                negativeSamples = produceNegativeSamples(distribution, numNegativeSamples, len(yb))
                if CUDA:
                    negativeSamples = negativeSamples.to('cuda')
                loss = model(yb, xb, negativeSamples)
                loss += torch.mean(loss).item()
            numBatchesProcessed += 1
            if numBatchesProcessed % args.batchesForLogging == 0:
                writeLog("Processed " + str(numBatchesProcessed) + " batches out of " + str(len(testDl)) +
                         " (testing)", logObject)
        loss = loss / len(testDl)
    writeLog("Test loss: " + str(loss), logObject)
    seconds = (datetime.now() - now).total_seconds()
    writeLog("Took: " + str(seconds) + " seconds to compute test loss", logObject)
    return loss


def writeLog(message, logObject, timestamp=datetime.now()):
    logObject.info("[" + str(timestamp) + "]: " + message)
    return


logger = logging.getLogger()
logging.basicConfig(level=logging.INFO, stream=stdout)
logger.addHandler(logging.FileHandler("log" + FULL_NAME + ".txt"))
writeLog("Running " + FULL_NAME, logger)
writeLog(str(args), logger)
if CUDA:
    writeLog("Cuda is available", logger)
else:
    writeLog("Cuda is not available", logger)

# Example usage:

wordIndex, reverseWordIndex, vocab, VOCAB_SIZE, wordFrequencies, sampleDistribution, trainDataLoader, validDataLoader, testDataLoader = setup('reviews_Grocery_and_Gourmet_Food_5.json.gz', logger, algorithm=args.algorithmType)
trainedModel = train(args.name, trainDataLoader, validDataLoader, VOCAB_SIZE, logger, distribution=sampleDistribution,
                     algorithm=args.algorithmType)
testLoss = finalEvaluation(trainedModel, testDataLoader, logger, distribution=sampleDistribution)
saveModelState(trainedModel, wordIndex, reverseWordIndex, vocab, wordFrequencies, logger)
wordIndex, reverseWordIndex, vocab, wordFrequencies, sampleDistribution, loadedModel = loadModelState(args.name, logger,
                                                                                                      algorithm=args.algorithmType)
topSimilarities = topKSimilarities(loadedModel, args.wordForComparison, wordIndex, vocab, logger,
                                   args.numWordsForComparison)
topSimilaritiesAnalogy = topKSimilaritiesAnalogy(loadedModel, args.word1ForAnalogy, args.word2ForAnalogy,
                                                 args.word3ForAnalogy, wordIndex, vocab, logger,
                                                 args.numWordsForComparison)

writeLog("Finished running " + FULL_NAME, logger)
