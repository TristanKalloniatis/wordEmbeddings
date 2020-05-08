import logging
from sys import stdout
import gzip
from json import loads
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
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
BATCH_SIZE = 1024
EPOCHS = 50  # todo: probably increase this once it looks like something is happening
LEARNING_RATE_DECAY_FACTOR = 0.5
PATIENCE = 2
UNIGRAM_DISTRIBUTION_POWER = 0.75
UNKNOWN_TOKEN = '???'
HIDDEN_STATE_CLAMP = 4
EXPLOSION_DETECTION_FACTOR = 3
MAX_EXPLOSIONS = 2
NUM_HIDDEN_STATES = 10
NUM_CATEGORIES = 2
MAX_REVIEW_LENGTH = 200
NUM_BATCHES_TO_PLOT = 1
BATCHES = EPOCHS * NUM_BATCHES_TO_PLOT

REVIEW_FILE = 'reviews_Grocery_and_Gourmet_Food_5.json.gz'
IMPLEMENTED_MODELS = ['CBOW', 'SGNS']
MIN_REVIEW_LENGTH = 2 * CONTEXT_SIZE + 1
FULL_NAME = 'RNN_HSC{0}_EDF{1}_ME{2}_NHS{3}_NC{4}_MRL{5}_{6}'.format(HIDDEN_STATE_CLAMP, EXPLOSION_DETECTION_FACTOR,
                                                                     MAX_EXPLOSIONS, NUM_HIDDEN_STATES, NUM_CATEGORIES,
                                                                     MAX_REVIEW_LENGTH, ALGORITHM_TYPE)


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
    writeLog("Loaded model {0}".format(modelName), logObject)
    model.eval()
    return wordMapping, reverseWordMapping, vocabulary, frequencies, distribution, model


def writeLog(message, logObject):
    timestamp = datetime.now()
    logObject.info("[{0}]: {1}".format(str(timestamp), message))
    return


class RNN(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize, learningRate):
        super().__init__()
        self.hiddenSize = hiddenSize
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.toHidden = nn.Linear(inputSize + hiddenSize, hiddenSize)
        self.toOutput = nn.Linear(hiddenSize, outputSize)
        self.trainLosses = []
        self.validLosses = []
        self.trainingTimes = []
        self.initialLearningRate = learningRate

    def forward(self, inputs):
        hidden = torch.zeros(self.hiddenSize)
        numWordsProcessed = 0
        for word in inputs:
            combined = torch.cat((word, hidden), dim=0)
            hidden = torch.clamp(self.toHidden(combined), min=-HIDDEN_STATE_CLAMP, max=HIDDEN_STATE_CLAMP)
            hidden = torch.tanh(hidden)
            numWordsProcessed += 1
            if numWordsProcessed >= MAX_REVIEW_LENGTH:
                break
        preOutput = torch.clamp(self.toOutput(hidden), min=-HIDDEN_STATE_CLAMP, max=HIDDEN_STATE_CLAMP)
        return F.log_softmax(preOutput, dim=-1)

    @property
    def numBatchesTrained(self):
        return len(self.trainLosses)


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
        if 'overall' not in review or 'reviewText' not in review:
            continue
        if review['overall'] not in [1.0, 2.0, 3.0, 4.0, 5.0]:
            continue
        words = preProcess(review['reviewText']).split()
        reviewLength = len(words)
    if review['overall'] == 5.0:
        targetOutput = 1
    else:
        targetOutput = 0
    if not embeddingModel:
        inputWords = [wordToTensor(word, vocabSize) for word in words]
    else:
        with torch.no_grad():
            inputWords = [embeddingModel.embeddings(torch.tensor(wordMapping[word.lower()], dtype=torch.long)) for word
                          in words]
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


def trainRNN(modelName, trainData, validData, algorithm, wordMapping, vocabSize, logObject, embeddingModel=None,
             embeddingDim=EMBEDDING_DIM, contextSize=CONTEXT_SIZE, numHiddenStates=NUM_HIDDEN_STATES,
             numCategories=NUM_CATEGORIES, batchSize=BATCH_SIZE, lr=LEARNING_RATE,
             learningRateDecayFactor=LEARNING_RATE_DECAY_FACTOR, patience=PATIENCE, batches=BATCHES,
             explosionDetectionFactor=EXPLOSION_DETECTION_FACTOR, maxExplosions=MAX_EXPLOSIONS):
    if algorithm:
        rnn = RNN(embeddingDim, numHiddenStates, numCategories, lr)
    else:
        rnn = RNN(vocabSize, numHiddenStates, numCategories, lr)
    rnnOptimiser = Adam(rnn.parameters(), lr=lr)
    rnnScheduler = ReduceLROnPlateau(rnnOptimiser, mode='min', factor=learningRateDecayFactor, patience=patience,
                                     verbose=True)
    trainLossesByBatch = []
    validLossesByBatch = []
    trainLossesByEpoch = []
    validLossesByEpoch = []
    batch = 0
    lastBatchExploded = False
    trainLossThisEpoch = 0
    validLossThisEpoch = 0
    while batch < batches:
        if not lastBatchExploded:
            numExplosionsThisBatch = 0
        now = datetime.now()
        writeLog("Running batch {0}".format(batch), logObject)
        rnnPreExplosion = rnn
        trainLoss = trainOnBatch(rnn, trainData, rnnOptimiser, wordMapping, vocabSize, batchSize, embeddingModel)
        validLoss = validateOnBatch(rnn, validData, wordMapping, vocabSize, batchSize, embeddingModel)
        seconds = (datetime.now() - now).total_seconds()
        rnn.trainingTimes.append(seconds)
        writeLog("Batch took: {0} seconds".format(str(seconds)), logObject)
        if batch > 0:
            if trainLoss > explosionDetectionFactor * trainLossesByBatch[batch - 1] \
                    or validLoss > explosionDetectionFactor * validLossesByBatch[batch - 1]:
                writeLog("Losses are exploding, aborting batch", logObject)
                numExplosionsThisBatch += 1
                lastBatchExploded = True
                rnn = rnnPreExplosion
                if numExplosionsThisBatch > maxExplosions:
                    writeLog("Exceeded the maximum number of explosions this batch, aborting training", logObject)
                    fig, ax = plt.subplots()
                    ax.plot(range(batch), trainLossesByBatch, label="Training")
                    ax.plot(range(batch), validLossesByBatch, label="Validation")
                    ax.set_xlabel("Batch")
                    ax.set_ylabel("Loss")
                    ax.set_title("Learning curve for RNN model {0}".format(modelName))
                    ax.legend()
                    plt.savefig(
                        '{0}learningCurveRNN{1}{2}{3}.png'.format(modelName, embeddingDim, algorithm, contextSize))
                    return rnn, trainLossesByEpoch, validLossesByEpoch
            else:
                trainLossesByBatch.append(trainLoss)
                validLossesByBatch.append(validLoss)
                trainLossThisEpoch += trainLoss
                validLossThisEpoch += validLoss
                rnn.trainLosses.append(trainLoss)
                rnn.validLosses.append(validLoss)
                writeLog("Training loss: {0}".format(trainLoss), logObject)
                writeLog("Validation loss: {0}".format(validLoss), logObject)
                batch += 1
                lastBatchExploded = False
                if batch % NUM_BATCHES_TO_PLOT == 0:
                    trainLossThisEpoch /= NUM_BATCHES_TO_PLOT
                    validLossThisEpoch /= NUM_BATCHES_TO_PLOT
                    trainLossesByEpoch.append(trainLossThisEpoch)
                    validLossesByEpoch.append(validLossThisEpoch)
                    writeLog("Average training loss over epoch {0} was {1}".format(len(trainLossesByEpoch),
                                                                                   trainLossThisEpoch),
                             logObject)
                    writeLog("Average validation loss over epoch {0} was {1}".format(len(validLossesByEpoch),
                                                                                     validLossThisEpoch),
                             logObject)
                    rnnScheduler.step(trainLossThisEpoch)  # todo: change this back to validLossThisEpoch
                    trainLossThisEpoch = 0
                    validLossThisEpoch = 0
        else:
            trainLossesByBatch.append(trainLoss)
            validLossesByBatch.append(validLoss)
            trainLossThisEpoch += trainLoss
            validLossThisEpoch += validLoss
            writeLog("Training loss: {0}".format(trainLoss), logObject)
            writeLog("Validation loss: {0}".format(validLoss), logObject)
            batch += 1
            lastBatchExploded = False
    _, ax = plt.subplots()
    ax.plot(range(batches), trainLossesByBatch, label="Training")
    ax.plot(range(batches), validLossesByBatch, label="Validation")
    ax.set_xlabel("Batch")
    ax.set_ylabel("Loss")
    ax.set_title("Learning curve for RNN model {0}".format(modelName))
    ax.legend()
    plt.savefig('{0}learningCurveRNN{1}{2}{3}.png'.format(modelName, embeddingDim, algorithm, contextSize))
    return rnn, trainLossesByEpoch, validLossesByEpoch


logger = logging.getLogger()
logging.basicConfig(level=logging.INFO, stream=stdout)
logger.addHandler(logging.FileHandler("log" + FULL_NAME + ".txt"))
writeLog("Running {0}".format(FULL_NAME), logger)

wordIndex, reverseWordIndex, vocab, wordFrequencies, sampleDistribution, loadedModel = loadModelState(NAME, logger,
                                                                                                      algorithm=
                                                                                                      ALGORITHM_TYPE)
if not ALGORITHM_TYPE:
    loadedModel = None
allRawData = getData(REVIEW_FILE, logger)
trainRawData, validRawData, testRawData = splitData(allRawData, TRAIN_PROPORTION, VALID_PROPORTION)
trainRawData = trainRawData[:NUM_BATCHES_TO_PLOT * BATCH_SIZE]
validRawData = validRawData[:NUM_BATCHES_TO_PLOT * BATCH_SIZE]
writeLog("{0} training reviews, {1} validation reviews, {2} testing reviews".format(len(trainRawData),
                                                                                    len(validRawData),
                                                                                    len(testRawData)),
         logger)
rnnModel, trainByEpoch, validByEpoch = trainRNN(NAME, trainRawData, validRawData, ALGORITHM_TYPE, wordIndex, len(vocab),
                                                logger, embeddingModel=loadedModel)
writeLog("Training losses by epoch:", logger)
for loss in trainByEpoch:
    writeLog(str(loss), logger)
writeLog("Validation losses by epoch:", logger)
for loss in validByEpoch:
    writeLog(str(loss), logger)


_, axes = plt.subplots()
axes.plot(range(EPOCHS - 1), trainByEpoch, label="Training")
axes.plot(range(EPOCHS - 1), validByEpoch, label="Validation")
axes.set_xlabel("Epoch")
axes.set_ylabel("Average loss")
axes.set_title("Learning curve for RNN model by epoch")
axes.legend()


def accuracy(data, rnn, logObject):
    preds = []
    actuals = []
    numCorrect = 0
    for review in data:
        words = preProcess(review['reviewText']).split()
        if len(words) < MIN_REVIEW_LENGTH:
            continue
        if 'overall' not in review:
            continue
        if review['overall'] == 5.0:
            actuals.append(1)
        else:
            actuals.append(0)
        with torch.no_grad():
            inputWords = [loadedModel.embeddings(torch.tensor(wordIndex[word.lower()], dtype=torch.long)) for word in
                          words]
            rnnOutput = rnn(inputWords)
            preds.append(torch.argmax(rnnOutput).item())
    writeLog("Number predicted: {0}, number actual: {1} (out of {2})".format(sum(preds), sum(actuals), len(actuals)),
             logObject)
    for i in range(len(actuals)):
        if preds[i] == actuals[i]:
            numCorrect += 1
    writeLog("Proportion correct: {0}".format(numCorrect / len(actuals)), logObject)
    return


writeLog("Train data accuracy", logger)
accuracy(trainRawData, rnnModel, logger)
writeLog("Valid data accuracy", logger)
accuracy(validRawData, rnnModel, logger)

writeLog("Finished running {0}".format(FULL_NAME), logger)
