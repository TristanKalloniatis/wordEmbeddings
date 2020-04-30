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
from random import randint
from math import sqrt

ALGORITHM_TYPE = 'CBOW'
NAME = 'groceries'
CONTEXT_SIZE = 3
EMBEDDING_DIM = 16
TRAIN_PROPORTION = 0.75
VALID_PROPORTION = 0.15
LEARNING_RATE = 0.005
MOMENTUM = 0.9
BATCH_SIZE = 1000
EPOCHS = 200
LEARNING_RATE_DECAY_FACTOR = 0.5
PATIENCE = 5
UNIGRAM_DISTRIBUTION_POWER = 0.75
UNKNOWN_TOKEN = '???'
HIDDEN_STATE_CLAMP = 10
EXPLOSION_DETECTION_FACTOR = 3
MAX_EXPLOSIONS = 2
NUM_HIDDEN_STATES = 100
NUM_CATEGORIES = 2

REVIEW_FILE = 'reviews_Grocery_and_Gourmet_Food_5.json.gz'
IMPLEMENTED_MODELS = ['CBOW', 'SGNS']
MIN_REVIEW_LENGTH = 2 * CONTEXT_SIZE + 1
CUDA = torch.cuda.is_available()
FULL_NAME = 'RNN' + 'HSC' + str(HIDDEN_STATE_CLAMP) + 'EDF' + str(EXPLOSION_DETECTION_FACTOR) + 'ME' + \
            str(MAX_EXPLOSIONS) + 'NHS' + str(NUM_HIDDEN_STATES) + 'NC' + str(NUM_CATEGORIES) + ALGORITHM_TYPE


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


def loadModelState(modelName, logObject, algorithm=ALGORITHM_TYPE, unigramDistributionPower=UNIGRAM_DISTRIBUTION_POWER):
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


def writeLog(message, logObject, timestamp=datetime.now()):
    logObject.info("[{0}]: {1}".format(str(timestamp), message))
    return


class RNN(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super().__init__()
        self.hiddenSize = hiddenSize
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.toHidden = nn.Linear(inputSize + hiddenSize, hiddenSize)
        self.toOutput = nn.Linear(inputSize + hiddenSize, outputSize)

    def forward(self, inputs):
        hidden = torch.zeros(self.hiddenSize)
        for word in inputs:
            combined = torch.cat((word, hidden), dim=0)
            hidden = torch.clamp(self.toHidden(combined), min=-HIDDEN_STATE_CLAMP, max=HIDDEN_STATE_CLAMP)
            preOutput = self.toOutput(combined)
        return F.log_softmax(preOutput, dim=-1)


def wordToTensor(word, vocabSize):
    tensor = torch.zeros(vocabSize)
    tensor[wordIndex[word]] = 1
    return tensor


def randomChoice(allReviews):
    index = randint(0, len(allReviews) - 1)
    return allReviews[index]


def randomTrainingExample(reviews, wordMapping, vocabSize, embeddingModel=None):
    reviewLength = 0
    while reviewLength < MIN_REVIEW_LENGTH:
        review = randomChoice(reviews)
        if 'overall' not in review:
            continue
        if review['overall'] not in [1.0, 2.0, 3.0, 4.0, 5.0]:
            continue
        words = preProcess(review['reviewText']).split()
        reviewLength = len(words)
    if review['overall'] == 5.0:
        targetOutput = 1
    else:
        targetOutput = 0
    inputWords = []
    if not embeddingModel:
        for word in words:
            wordEmbedding = wordToTensor(word, vocabSize)
            inputWords.append(wordEmbedding)
    else:
        with torch.no_grad():
            for word in words:
                wordTensor = torch.tensor(wordMapping[word.lower()], dtype=torch.long)
                wordEmbedding = embeddingModel.embeddings(wordTensor)
                inputWords.append(wordEmbedding)
    return inputWords, targetOutput


def trainOnExample(targetTensor, inputWords, rnn, criterion=nn.NLLLoss()):
    output = rnn(inputWords)
    lossOnElement = criterion(output.unsqueeze(0), targetTensor)
    return lossOnElement


def trainOnBatch(rnn, reviews, rnnOptimiser, wordMapping, vocabSize, batchSize=BATCH_SIZE, embeddingModel=None):
    batchLoss = 0
    rnnOptimiser.zero_grad()
    for _ in range(batchSize):
        inputWords, target = randomTrainingExample(reviews, wordMapping, vocabSize, embeddingModel)
        lossOnElement = trainOnExample(torch.tensor([target], dtype=torch.long), inputWords, rnn)
        batchLoss += lossOnElement
    batchLoss = batchLoss / batchSize
    batchLoss.backward()
    rnnOptimiser.step()
    return batchLoss.item()


def validateOnBatch(rnn, reviews, wordMapping, vocabSize, batchSize=BATCH_SIZE, embeddingModel=None):
    batchLoss = 0
    with torch.no_grad():
        for _ in range(batchSize):
            inputWords, target = randomTrainingExample(reviews, wordMapping, vocabSize, embeddingModel)
            lossOnElement = trainOnExample(torch.tensor([target], dtype=torch.long), inputWords, rnn)
            batchLoss += lossOnElement
    batchLoss = batchLoss / batchSize
    return batchLoss.item()


def trainRNN(modelName, epochs, trainData, validData, algorithm, wordMapping, vocabSize, logObject, embeddingModel=None,
             embeddingDim=EMBEDDING_DIM, contextSize=CONTEXT_SIZE, numHiddenStates=NUM_HIDDEN_STATES,
             numCategories=NUM_CATEGORIES, batchSize=BATCH_SIZE, lr=LEARNING_RATE, momentum=MOMENTUM,
             learningRateDecayFactor=LEARNING_RATE_DECAY_FACTOR, patience=PATIENCE,
             explosionDetectionFactor=EXPLOSION_DETECTION_FACTOR, maxExplosions=MAX_EXPLOSIONS):
    if algorithm:
        rnn = RNN(embeddingDim, numHiddenStates, numCategories)
    else:
        rnn = RNN(vocabSize, numHiddenStates, numCategories)
    rnnOptimiser = SGD(rnn.parameters(), lr=lr, momentum=momentum, nesterov=True)
    rnnScheduler = ReduceLROnPlateau(rnnOptimiser, mode='min', factor=learningRateDecayFactor, patience=patience,
                                     verbose=True)
    trainLosses = []
    validLosses = []
    epoch = 0
    lastEpochExploded = False
    while epoch < epochs:
        if not lastEpochExploded:
            numExplosionsThisEpoch = 0
        now = datetime.now()
        writeLog("Running epoch {0}".format(epoch), logObject)
        rnnPreExplosion = rnn
        trainLoss = trainOnBatch(rnn, trainData, rnnOptimiser, wordMapping, vocabSize, batchSize, embeddingModel)
        validLoss = validateOnBatch(rnn, validData, wordMapping, vocabSize, batchSize, embeddingModel)
        seconds = (datetime.now() - now).total_seconds()
        writeLog("Epoch took: {0} seconds".format(str(seconds)), logObject)
        if epoch > 0:
            if trainLoss > explosionDetectionFactor * trainLosses[epoch - 1] \
                    or validLoss > explosionDetectionFactor * validLosses[epoch - 1]:
                writeLog("Losses are exploding, aborting epoch", logObject)
                numExplosionsThisEpoch += 1
                lastEpochExploded = True
                rnn = rnnPreExplosion
                if numExplosionsThisEpoch > maxExplosions:
                    writeLog("Exceeded the maximum number of explosions this epoch, aborting training", logObject)
                    fig, ax = plt.subplots()
                    ax.plot(range(epoch), trainLosses, label="Training")
                    ax.plot(range(epoch), validLosses, label="Validation")
                    ax.set_xlabel("Epoch")
                    ax.set_ylabel("Loss")
                    ax.set_title("Learning curve for RNN model {0}".format(modelName))
                    ax.legend()
                    plt.savefig(
                        '{0}learningCurveRNN{1}{2}{3}.png'.format(modelName, embeddingDim, algorithm, contextSize))
                    return rnn
            else:
                trainLosses.append(trainLoss)
                validLosses.append(validLoss)
                writeLog("Training loss: {0}".format(trainLoss), logObject)
                writeLog("Validation loss: {0}".format(validLoss), logObject)
                epoch += 1
                lastEpochExploded = False
                rnnScheduler.step(validLoss)
        else:
            trainLosses.append(trainLoss)
            validLosses.append(validLoss)
            writeLog("Training loss: {0}".format(trainLoss), logObject)
            writeLog("Validation loss: {0}".format(validLoss), logObject)
            epoch += 1
            lastEpochExploded = False
            rnnScheduler.step(validLoss)
    fig, ax = plt.subplots()
    ax.plot(range(epochs), trainLosses, label="Training")
    ax.plot(range(epochs), validLosses, label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Learning curve for RNN model {0}".format(modelName))
    ax.legend()
    plt.savefig('{0}learningCurveRNN{1}{2}{3}.png'.format(modelName, embeddingDim, algorithm, contextSize))
    return rnn


logger = logging.getLogger()
logging.basicConfig(level=logging.INFO, stream=stdout)
logger.addHandler(logging.FileHandler("log" + FULL_NAME + ".txt"))
writeLog("Running {0}".format(FULL_NAME), logger)
if CUDA:
    writeLog("Cuda is available", logger)
else:
    writeLog("Cuda is not available", logger)

wordIndex, reverseWordIndex, vocab, wordFrequencies, sampleDistribution, loadedModel = loadModelState(NAME, logger,
                                                                                                      algorithm=
                                                                                                      ALGORITHM_TYPE)
if not ALGORITHM_TYPE:
    loadedModel = None
allRawData = getData(REVIEW_FILE, logger)
trainRawData, validRawData, testRawData = splitData(allRawData, TRAIN_PROPORTION, VALID_PROPORTION)
writeLog("{0} training reviews, {1} validation reviews, {2} testing reviews".format(len(trainRawData),
                                                                                    len(validRawData),
                                                                                    len(testRawData)),
         logger)

writeLog("Finished running {0}".format(FULL_NAME), logger)
