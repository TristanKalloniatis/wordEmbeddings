import logging
from sys import stdout
import gzip
from json import loads
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from pickle import dump, load
from math import sqrt
from torch.utils.data import DataLoader, TensorDataset

NAME = 'groceries'
TRAIN_PROPORTION = 0.75
VALID_PROPORTION = 0.15
LEARNING_RATE = 1
MOMENTUM = 0.9
BATCH_SIZE = 1000
EPOCHS = 30
LEARNING_RATE_DECAY_FACTOR = 0.1
PATIENCE = 5
UNKNOWN_TOKEN = '???'
NUM_CATEGORIES = 2
MIN_WORD_COUNT = 10
DF_SCORE_THRESHOLD = 1e-6

REVIEW_FILE = 'reviews_Grocery_and_Gourmet_Food_5.json.gz'
IMPLEMENTED_MODELS = ['CBOW', 'SGNS']
MIN_REVIEW_LENGTH = 7
CUDA = torch.cuda.is_available()
FULL_NAME = 'LR' + 'NC' + str(NUM_CATEGORIES)


def noiseDistribution(frequencies, unigramDistributionPower):
    adjustedFrequencies = [frequency ** unigramDistributionPower for frequency in frequencies]
    normalisation = sum(adjustedFrequencies)
    return [adjustedFrequency / normalisation for adjustedFrequency in adjustedFrequencies]


def getData(filePath, logObject):
    file = gzip.open(filePath, mode='rb')
    rawData = []
    for line in file:
        rawData.append(loads(line))
    file.close()
    writeLog("Number of reviews: {0}".format(str(len(rawData))), logObject)
    return rawData


def preProcess(text):
    text = text.lower()
    result = ''
    for x in text:
        if x.isalpha() or x == ' ':
            result += x
    return result


def splitData(rawData, trainProportion, validProportion):
    trainData = rawData[: int(len(rawData) * trainProportion)]
    validData = rawData[int(len(rawData) * trainProportion):int(len(rawData) * (trainProportion + validProportion))]
    testData = rawData[int(len(rawData) * (trainProportion + validProportion)):]
    return trainData, validData, testData


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


class LogisticRegression(nn.Module):
    def __init__(self, inputSize, outputSize):
        super().__init__()
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.linear = nn.Linear(self.inputSize, self.outputSize)

    def forward(self, inputs):
        out = self.linear(inputs)
        return F.log_softmax(out, dim=-1)


def loadModelState(modelName, logObject, algorithm, unigramDistributionPower=0.75):
    infile = open(modelName + 'WordMapping', 'rb')
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
    writeLog("Loaded model {0}".format(modelName), logObject)
    model.eval()
    return wordMapping, reverseWordMapping, vocabulary, frequencies, distribution, model


def writeLog(message, logObject):
    timestamp = datetime.now()
    logObject.info("[{0}]: {1}".format(str(timestamp), message))
    return


def buildDataLoader(rawData, wordMapping, vocabSize, logObject, minReviewLength=MIN_REVIEW_LENGTH, batchSize=None,
                    shuffle=False):
    xs = []
    ys = []
    for review in rawData:
        words = preProcess(review['reviewText']).split()
        if len(words) < minReviewLength:
            continue
        xs.append(makeBOWVector(words, wordMapping, vocabSize))
        if review['overall'] == 5.0:
            ys.append(1)
        else:
            ys.append(0)
    writeLog("Size of data: {0}".format(len(xs)), logObject)
    xs, ys = map(torch.tensor, (xs, ys))
    ds = TensorDataset(xs, ys)
    if batchSize is not None:
        dl = DataLoader(ds, batch_size=batchSize, shuffle=shuffle)
    else:
        dl = DataLoader(ds, shuffle=shuffle)
    return dl


def buildDataLoaderEmbeddingModel(rawData, wordMapping, embeddingModel, logObject,
                                  minReviewLength=MIN_REVIEW_LENGTH, batchSize=None, shuffle=False):
    xs = []
    ys = []
    for review in rawData:
        words = preProcess(review['reviewText']).split()
        if len(words) < minReviewLength:
            continue
        xs.append(makeEmbedVector(words, wordMapping, embeddingModel))
        if review['overall'] == 5.0:
            ys.append(1)
        else:
            ys.append(0)
    writeLog("Size of data: {0}".format(len(xs)), logObject)
    xs, ys = map(torch.tensor, (xs, ys))
    ds = TensorDataset(xs, ys)
    if batchSize is not None:
        dl = DataLoader(ds, batch_size=batchSize, shuffle=shuffle)
    else:
        dl = DataLoader(ds, shuffle=shuffle)
    return dl


def makeBOWVector(words, wordMapping, vocabSize):
    vector = [0.] * vocabSize
    for word in words:
        vector[wordMapping[word.lower()]] += 1 / len(words)
    return vector


def makeEmbedVector(words, wordMapping, embeddingModel):
    vector = torch.zeros(embeddingModel.embeddingDim)
    with torch.no_grad():
        for word in words:
            wordTensor = torch.tensor(wordMapping[word.lower()], dtype=torch.long)
            wordEmbedding = embeddingModel.embeddings(wordTensor)
            vector = vector + wordEmbedding
    return vector.numpy() / len(words)


def buildWordCounts(rawData, logObject):
    wordCounts = {}
    for review in rawData:
        words = preProcess(review['reviewText']).split()
        for word in words:
            if word in wordCounts:
                wordCounts[word] += 1
            else:
                wordCounts[word] = 1
    writeLog("Number of distinct words: {0}".format(len(wordCounts)), logObject)
    return wordCounts


def buildVocab(rawData, minWordCount, unknownToken, logObject):
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
        writeLog("Words exist with total count less than {0} which will be replaced with {1}".format(minWordCount,
                                                                                                     unknownToken),
                 logObject)
        reverseWordMapping[len(allowableVocab)] = unknownToken
        frequencies.append(totalRareWords / numWords)
        wordMapping[unknownToken] = len(allowableVocab)
        for word in wordCounts:
            if wordCounts[word] < minWordCount:
                wordMapping[word] = len(allowableVocab)
        allowableVocab.append(unknownToken)
    vocabularySize = len(allowableVocab)
    writeLog("Vocabulary size: {0}".format(vocabularySize), logObject)
    return wordMapping, reverseWordMapping, allowableVocab, vocabularySize, frequencies


def setupEmbed(filePath, embeddingModel, logObject, batchSize=BATCH_SIZE, minWordCount=MIN_WORD_COUNT,
               unknownToken=UNKNOWN_TOKEN, trainProportion=TRAIN_PROPORTION, validProportion=VALID_PROPORTION):
    now = datetime.now()
    data = getData(filePath, logObject)
    wordMapping, reverseWordMapping, allowableVocab, vocabSize, frequencies = buildVocab(data, minWordCount,
                                                                                         unknownToken,
                                                                                         logObject)
    trainData, validData, testData = splitData(data, trainProportion, validProportion)
    writeLog("Train data", logObject)
    trainDl = buildDataLoaderEmbeddingModel(trainData, wordMapping, embeddingModel, logObject, batchSize=batchSize,
                                            shuffle=True)
    writeLog("Validation data", logObject)
    validDl = buildDataLoaderEmbeddingModel(validData, wordMapping, embeddingModel, logObject, batchSize=2 * batchSize,
                                            shuffle=False)
    writeLog("Test data", logObject)
    testDl = buildDataLoaderEmbeddingModel(testData, wordMapping, embeddingModel, logObject, batchSize=2 * batchSize,
                                          shuffle=False)
    seconds = (datetime.now() - now).total_seconds()
    writeLog("Setting up took: {0} seconds".format(seconds), logObject)
    return wordMapping, reverseWordMapping, allowableVocab, vocabSize, frequencies, trainDl, validDl, testDl


def setup(filePath, logObject, batchSize=BATCH_SIZE, minWordCount=MIN_WORD_COUNT, unknownToken=UNKNOWN_TOKEN,
          trainProportion=TRAIN_PROPORTION, validProportion=VALID_PROPORTION):
    now = datetime.now()
    data = getData(filePath, logObject)
    wordMapping, reverseWordMapping, allowableVocab, vocabSize, frequencies = buildVocab(data, minWordCount,
                                                                                         unknownToken,
                                                                                         logObject)
    trainData, validData, testData = splitData(data, trainProportion, validProportion)
    writeLog("Train data", logObject)
    trainDl = buildDataLoader(trainData, wordMapping, vocabSize, logObject, batchSize=batchSize, shuffle=True)
    writeLog("Validation data", logObject)
    validDl = buildDataLoader(validData, wordMapping, vocabSize, logObject, batchSize=2 * batchSize, shuffle=False)
    writeLog("Test data", logObject)
    testDl = buildDataLoader(testData, wordMapping, vocabSize, logObject, batchSize=2 * batchSize, shuffle=False)
    seconds = (datetime.now() - now).total_seconds()
    writeLog("Setting up took: {0} seconds".format(seconds), logObject)
    return wordMapping, reverseWordMapping, allowableVocab, vocabSize, frequencies, trainDl, validDl, testDl


def train(modelName, trainDl, validDl, logObject, useEmbeds=False, epochs=EPOCHS, lr=LEARNING_RATE, momentum=MOMENTUM,
          learningRateDecayFactor=LEARNING_RATE_DECAY_FACTOR, patience=PATIENCE):
    trainLosses = []
    valLosses = []
    model = LogisticRegression(VOCAB_SIZE, NUM_CATEGORIES)
    if useEmbeds:
        model = LogisticRegression(16, NUM_CATEGORIES)
    lossFunction = nn.NLLLoss()
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=True)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=learningRateDecayFactor, patience=patience,
                                  verbose=True)

    for epoch in range(epochs):
        now = datetime.now()
        writeLog("Epoch: {0}".format(epoch), logObject)
        writeLog("Training on {0} batches and validating on {1} batches".format(len(trainDl), len(validDl)), logObject)

        model.train()
        totalLoss = 0
        numBatchesProcessed = 0
        for xb, yb in trainDl:
            predictions = model(xb)
            loss = lossFunction(predictions, yb)
            loss.backward()
            totalLoss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            numBatchesProcessed += 1
            if numBatchesProcessed % 1000 == 0:
                writeLog(
                    "Processed {0} batches out of {1} (training)".format(numBatchesProcessed, len(trainDl)), logObject)
        trainLoss = totalLoss / len(trainDl)
        writeLog("Training loss: {0}".format(trainLoss), logObject)
        trainLosses.append(trainLoss)

        model.eval()
        with torch.no_grad():
            validLoss = 0
            numBatchesProcessed = 0
            for xb, yb in validDl:
                validLoss += lossFunction(model(xb), yb).item()
                numBatchesProcessed += 1
                if numBatchesProcessed % 1000 == 0:
                    writeLog("Processed {0} batches out of {1} (validation)".format(numBatchesProcessed, len(validDl)),
                             logObject)
        validLoss = validLoss / len(validDl)
        valLosses.append(validLoss)
        writeLog("Validation loss: {0}".format(validLoss), logObject)

        seconds = (datetime.now() - now).total_seconds()
        writeLog("Epoch took: {0} seconds".format(seconds), logObject)
        scheduler.step(validLoss)

    fig, ax = plt.subplots()
    ax.plot(range(epochs), trainLosses, label="Training")
    ax.plot(range(epochs), valLosses, label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Learning curve for model {0}".format(modelName))
    ax.legend()
    plt.show()

    return model, trainLosses, valLosses


logger = logging.getLogger()
logging.basicConfig(level=logging.INFO, stream=stdout)
logger.addHandler(logging.FileHandler("log" + FULL_NAME + ".txt"))
writeLog("Running {0}".format(FULL_NAME), logger)
if CUDA:
    writeLog("Cuda is available", logger)
else:
    writeLog("Cuda is not available", logger)

writeLog("BOW", logger)
wordIndex, reverseWordIndex, vocab, VOCAB_SIZE, wordFrequencies, trainDataLoaderBOW, validDataLoaderBOW, testDataLoaderBOW = setup(
    REVIEW_FILE, logger)
trainedModelBOW, trainLossesBOW, validLossesBOW = train(NAME + 'BOW', trainDataLoaderBOW, validDataLoaderBOW, logger)
writeLog("CBOW", logger)
wordIndex, reverseWordIndex, vocab, wordFrequencies, sampleDistribution, loadedModel = loadModelState(NAME, logger,
                                                                                                      algorithm=
                                                                                                      'CBOW')
wordIndex, reverseWordIndex, vocab, VOCAB_SIZE, wordFrequencies, trainDataLoaderCBOW, validDataLoaderCBOW, testDataLoaderCBOW = setupEmbed(REVIEW_FILE, loadedModel, logger)
trainedModelCBOW, trainLossesCBOW, validLossesCBOW = train(NAME + 'CBOW', trainDataLoaderCBOW, validDataLoaderCBOW, logger, useEmbeds=True)
writeLog("SGNS", logger)
wordIndex, reverseWordIndex, vocab, wordFrequencies, sampleDistribution, loadedModel = loadModelState(NAME, logger,
                                                                                                      algorithm=
                                                                                                      'SGNS')
wordIndex, reverseWordIndex, vocab, VOCAB_SIZE, wordFrequencies, trainDataLoaderSGNS, validDataLoaderSGNS, testDataLoaderSGNS = setupEmbed(REVIEW_FILE, loadedModel, logger)
trainedModelSGNS, trainLossesSGNS, validLossesSGNS = train(NAME + 'SGNS', trainDataLoaderSGNS, validDataLoaderSGNS, logger, useEmbeds=True)
writeLog("Finished running {0}".format(FULL_NAME), logger)
