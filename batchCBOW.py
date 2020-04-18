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

CONTEXT_SIZE = 5
EMBEDDING_DIM = 16
MIN_WORD_COUNT = 10
TRAIN_PROPORTION = 0.75
VALID_PROPORTION = 0.15
LEARNING_RATE = 1
MOMENTUM = 0.9
BATCH_SIZE = 100
EPOCHS = 20
LEARNING_RATE_DECAY_FACTOR = 0.1
PATIENCE = 1
SUBSAMPLE_THRESHOLD = 1e-5
UNIGRAM_DISTRIBUTION_POWER = 0.75
NUM_NEGATIVE_SAMPLES = 10
UNKNOWN_TOKEN = '???'


def subsampleProbabilityDiscard(wordFrequency, threshold=SUBSAMPLE_THRESHOLD):
    if wordFrequency <= 0:
        return 0
    rawResult = 1 - sqrt(threshold / wordFrequency)
    if rawResult < 0:
        return 0
    return rawResult


def subsampleWord(word, wordFrequencies, threshold=SUBSAMPLE_THRESHOLD):
    return random() < subsampleProbabilityDiscard(wordFrequencies[word], threshold)


def noiseDistribution(frequencies, unigramDistributionPower=UNIGRAM_DISTRIBUTION_POWER):
    adjustedWordFrequencies = {frequencies[word] ** unigramDistributionPower for word in frequencies}
    normalisation = sum(adjustedWordFrequencies[word] for word in adjustedWordFrequencies)
    return {word: adjustedWordFrequencies[word] / normalisation for word in adjustedWordFrequencies}


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


def buildVocab(rawData, minWordCount=MIN_WORD_COUNT, unknownToken=UNKNOWN_TOKEN):
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
    frequencies = {word: wordCounts[word] / numWords for word in wordCounts}
    if totalRareWords > 0:
        reverseWordMapping[len(allowableVocab)] = unknownToken
        for word in wordCounts:
            if wordCounts[word] < minWordCount:
                wordMapping[word] = len(allowableVocab)
        allowableVocab.append(unknownToken)
        frequencies[unknownToken] = totalRareWords / numWords
    vocabularySize = len(allowableVocab)
    print("Vocabulary size:", vocabularySize)
    return wordMapping, reverseWordMapping, allowableVocab, vocabularySize, frequencies


def preProcessWords(words, wordMapping, contextSize=CONTEXT_SIZE, algorithm='CBOW'):
    dataPoints = []
    for i in range(contextSize, len(words) - contextSize):
        context = [wordMapping[words[i - j - 1]] for j in range(contextSize)]
        context += [wordMapping[words[i + j + 1]] for j in range(contextSize)]
        target = wordMapping[words[i]]
        if algorithm == 'CBOW':
            dataPoints.append((context, target))
        else:
            for word in context:
                dataPoints.append((word, target))
    return dataPoints


def splitData(rawData, trainProportion=TRAIN_PROPORTION, validProportion=VALID_PROPORTION):
    trainData = rawData[: int(len(rawData) * trainProportion)]
    validData = rawData[int(len(rawData) * trainProportion):int(len(rawData) * (trainProportion + validProportion))]
    testData = rawData[int(len(rawData) * (trainProportion + validProportion)):]
    return trainData, validData, testData


def buildDataLoader(rawData, wordMapping, reverseWordMapping, frequencies, subSample=False, batchSize=None,
                    shuffle=False, algorithm='CBOW'):
    xs = []
    ys = []
    for review in rawData:
        dataPoints = preProcessWords(preProcess(review['reviewText']).split(), wordMapping, algorithm)
        for dataPointX, dataPointY in dataPoints:
            if subSample:
                targetWord = reverseWordMapping[dataPointY]
                if subsampleWord(targetWord, frequencies):
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


def setup(filePath, batchSize=BATCH_SIZE, algorithm='CBOW'):
    data = getData(filePath)
    wordMapping, reverseWordMapping, allowableVocab, vocabularySize, frequencies = buildVocab(data)
    trainData, validData, testData = splitData(data)
    print("Train data")
    trainDl = buildDataLoader(trainData, wordMapping, reverseWordMapping, frequencies, subSample=True,
                              batchSize=batchSize, shuffle=True, algorithm=algorithm)
    print("Validation data")
    validDl = buildDataLoader(validData, wordMapping, reverseWordMapping, frequencies, subSample=True,
                              batchSize=2 * batchSize, algorithm=algorithm)
    print("Test data")
    testDl = buildDataLoader(testData, wordMapping, reverseWordMapping, frequencies, batchSize=2 * batchSize,
                             algorithm=algorithm)
    return wordMapping, reverseWordMapping, allowableVocab, vocabularySize, frequencies, trainDl, validDl, testDl


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
    def __init__(self, vocabSize, embeddingDim, contextSize, numNegativeSamples=NUM_NEGATIVE_SAMPLES):
        super().__init__()
        self.inEmbeddings = nn.Embedding(vocabSize, embeddingDim)
        self.outEmbeddings = nn.Embedding(vocabSize, embeddingDim)
        self.contextSize = contextSize
        self.embeddingDim = embeddingDim
        self.vocabSize = vocabSize
        self.numNegativeSamples = numNegativeSamples

    def forward(self, inputs):
        inputEmbedding = self.inEmbeddings(inputs)
        return 1


def train(trainDl, validDl, vocabSize, epochs=EPOCHS, embeddingDim=EMBEDDING_DIM, contextSize=CONTEXT_SIZE,
          lr=LEARNING_RATE, momentum=MOMENTUM, learningRateDecayFactor=LEARNING_RATE_DECAY_FACTOR, patience=PATIENCE,
          algorithm='CBOW'):
    trainLosses = []
    valLosses = []
    lossFunction = nn.NLLLoss()
    model = ContinuousBagOfWords(vocabSize, embeddingDim, contextSize)
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=True)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=learningRateDecayFactor, patience=patience,
                                  verbose=True)

    for epoch in range(epochs):
        print("Epoch:", epoch)
        now = datetime.now()

        model.train()
        totalLoss = 0
        for xb, yb in trainDl:
            if algorithm == 'CBOW':
                predictions = model(xb)
                loss = lossFunction(predictions, yb)
            else:
                # SG code here
                pass
            loss.backward()
            totalLoss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
        trainLoss = totalLoss / len(trainDl)
        print("Training loss:", trainLoss)
        trainLosses.append(trainLoss)

        model.eval()
        with torch.no_grad():
            if algorithm == 'CBOW':
                validLoss = sum(lossFunction(model(xb), yb) for xb, yb in validDl).item()
            else:
                # SG code here
                pass
        validLoss = validLoss / len(validDl)
        valLosses.append(validLoss)
        print("Validation loss:", validLoss)

        seconds = (datetime.now() - now).total_seconds()
        print("Took:", seconds)
        scheduler.step(validLoss)

    plt.plot(trainLosses, 'go--', linewidth=2, markersize=12)
    plt.plot(valLosses, 'bo--', linewidth=2, markersize=12)

    return model


def saveModelState(model, modelName, wordMapping, reverseWordMapping, vocabulary, frequencies, algorithm='CBOW'):
    torch.save(model.state_dict(), modelName + '.pt')
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
    if algorithm == 'CBOW':
        modelData = {'embeddingDim': model.embeddingDim, 'contextSize': model.contextSize}
    else:
        modelData = {'embeddingDim': model.embeddingDim, 'contextSize': model.contextSize,
                     'numNegativeSamples': model.numNegativeSamples}
    outfile = open(modelName + 'ModelData', 'wb')
    dump(modelData, outfile)
    outfile.close()


def loadModelState(modelName, algorithm='CBOW'):
    infile = open(modelName + 'wordMapping', 'rb')
    wordMapping = load(infile)
    infile.close()
    infile = open(modelName + 'reverseWordMapping', 'rb')
    reverseWordMapping = load(infile)
    infile.close()
    infile = open(modelName + 'Vocab', 'rb')
    vocab = load(infile)
    infile.close()
    infile = open(modelName + 'Frequencies', 'rb')
    frequencies = load(infile)
    infile.close()
    infile = open(modelName + 'ModelData', 'rb')
    modelData = load(infile)
    infile.close()
    if algorithm == 'CBOW':
        model = ContinuousBagOfWords(len(vocab), modelData['embeddingDim'], modelData['contextSize'])
    else:
        model = SkipGramWithNegativeSampling(len(vocab), modelData['embeddingDim'], modelData['contextSize'],
                                             modelData['numNegativeSamples'])
    model.load_state_dict(torch.load(modelName + '.pt'))
    model.eval()
    return wordMapping, reverseWordMapping, vocab, frequencies, model


def topKSimilarities(model, word, wordMapping, vocabulary, K=10):
    allSimilarities = {}
    with torch.no_grad():
        wordEmbedding = model.embeddings(torch.tensor(wordMapping[word], dtype=torch.long))
        for otherWord in vocabulary:
            otherEmbedding = model.embeddings(torch.tensor(wordMapping[otherWord], dtype=torch.long))
            allSimilarities[otherWord] = nn.CosineSimilarity(dim=0)(wordEmbedding, otherEmbedding).item()
    return {k: v for k, v in sorted(allSimilarities.items(), key=lambda item: item[1], reverse=True)[1:K + 1]}


def topKSimilaritiesAnalogy(model, word1, word2, word3, wordMapping, vocabulary, K=10):
    allSimilarities = {}
    with torch.no_grad():
        word1Embedding = model.embeddings(torch.tensor(wordMapping[word1], dtype=torch.long))
        word2Embedding = model.embeddings(torch.tensor(wordMapping[word2], dtype=torch.long))
        word3Embedding = model.embeddings(torch.tensor(wordMapping[word3], dtype=torch.long))
        diff = word1Embedding - word2Embedding + word3Embedding
        for otherWord in vocabulary:
            otherEmbedding = model.embeddings(torch.tensor(wordMapping[otherWord], dtype=torch.long))
            allSimilarities[otherWord] = nn.CosineSimilarity(dim=0)(diff, otherEmbedding).item()
    return {k: v for k, v in sorted(allSimilarities.items(), key=lambda item: item[1], reverse=True)[:K]}


def finalEvaluation(model, testDl, lossFunction=nn.NLLLoss(), algorithm='CBOW'):
    with torch.no_grad():
        if algorithm == 'CBOW':
            testLoss = sum(lossFunction(model(xb), yb).item() for xb, yb in testDl)
        else:
            # SG code here
            pass
        testLoss = testLoss / len(testDl)
    return testLoss

# Example usage:

# wordIndex, reverseWordIndex, vocab, VOCAB_SIZE, wordFrequencies, trainDl, validDl, testDl =
#                                                       setup('reviews_Grocery_and_Gourmet_Food_5.json.gz', 'CBOW')
# trainedModel = train(trainDl, validDl, vocabSize)
# print(finalEvaluation(trainedModel, testDl))
# saveModelState(trainedModel, 'groceriesCBOWSubSample', wordIndex, reverseWordIndex, vocab, wordFrequencies, 'CBOW')
# wordIndex, reverseWordIndex, vocab, wordFrequencies, loadedModel = loadModelState('groceriesCBOWSubSample', 'CBOW')
# print(topKSimilarities(loadedModel, 'apple', wordIndex, vocab))
# print(topKSimilaritiesAnalogy(loadedModel, 'buying', 'buy', 'sell', wordIndex, vocab))
