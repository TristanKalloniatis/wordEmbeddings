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

parser = ArgumentParser(description='Options for word2vec')
parser.add_argument("--contextSize", type=int, default=CONTEXT_SIZE, help="Context size for training")
parser.add_argument("--embeddingDimension", type=int, default=EMBEDDING_DIM, help="Internal embedding dimension")
parser.add_argument("--minWordCount", type=int, default=MIN_WORD_COUNT,
                    help="Minimum word count to not be mapped to unknown word")
parser.add_argument("--trainProportion", type=float, default=TRAIN_PROPORTION,
                    help="Proportion of reviews to use in training set")
parser.add_argument("--validProportion", type=float, default=VALID_PROPORTION,
                    help="Proportion of reviews to use in validation set")
parser.add_argument("--learningRate", type=float, default=LEARNING_RATE, help="Initial learning rate to use")
parser.add_argument("--momentum", type=float, default=MOMENTUM, help="Momentum to use in optimiser")
parser.add_argument("--batchSize", type=int, default=BATCH_SIZE,  help="Batch size for training")
parser.add_argument("--batchesForLogging", type=int, default=BATCHES_FOR_LOGGING,
                    help="After how many batches processed should the progress be logged")
parser.add_argument("--epochs", type=int, default=EPOCHS, help="How many epochs to train for")
parser.add_argument("--learningRateDecayFactor", type=float, default=LEARNING_RATE_DECAY_FACTOR,
                    help="How much to reduce the learning rate when plateauing")
parser.add_argument("--patience", type=int, default=PATIENCE,
                    help="How many epochs without progress until plateau is declared")
parser.add_argument("--subsampleThreshold", type=float, default=SUBSAMPLE_THRESHOLD,
                    help="Threshold frequency of words to begin subsampling")
parser.add_argument("--unigramDistributionPower", type=float, default=UNIGRAM_DISTRIBUTION_POWER,
                    help="Adjustment to unigram distribution to make in selecting negative samples")
parser.add_argument("--numNegativeSamples", type=int, default=NUM_NEGATIVE_SAMPLES,
                    help="Number of negative samples to use")
parser.add_argument("--innerProductClamp", type=float, default=INNER_PRODUCT_CLAMP,
                    help="How much to clamp the internal inner products")
parser.add_argument("--algorithmType", type=str, default='CBOW', help="Which algorithm to use")
parser.add_argument("--name", type=str, required=True, help="Name for the model")

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

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO, stream=stdout)
if CUDA:
    logger.info("Cuda is available")
else:
    logger.info("Cuda is not available")
logger.info("[" + str(datetime.now()) + "]: Running " + FULL_NAME)
logger.addHandler(logging.FileHandler("log" + FULL_NAME + ".txt"))


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
    logger.info("[" + str(datetime.now()) + "]: Number of reviews: " + str(len(rawData)))
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
    logger.info("[" + str(datetime.now()) + "]: Number of distinct words: " + str(len(wordCounts)))
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
        logger.info("[" + str(datetime.now()) + "]: Words exist with total count less than " + str(minWordCount) +
                    " which will be replaced with " + unknownToken)
        reverseWordMapping[len(allowableVocab)] = unknownToken
        frequencies.append(totalRareWords / numWords)
        wordMapping[unknownToken] = len(allowableVocab)
        for word in wordCounts:
            if wordCounts[word] < minWordCount:
                wordMapping[word] = len(allowableVocab)
        allowableVocab.append(unknownToken)
    vocabularySize = len(allowableVocab)
    distribution = noiseDistribution(frequencies, unigramDistributionPower)
    logger.info("[" + str(datetime.now()) + "]: Vocabulary size: " + str(vocabularySize))
    return wordMapping, reverseWordMapping, allowableVocab, vocabularySize, frequencies, distribution


def preProcessWords(words, wordMapping, contextSize, algorithm, minReviewLength):
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


def buildDataLoader(rawData, wordMapping, frequencies, contextSize, algorithm, threshold,
                    minReviewLength=MIN_REVIEW_LENGTH, subSample=False, batchSize=None, shuffle=False):
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
    logger.info("[" + str(datetime.now()) + "]: Size of data: " + str(len(xs)))
    xs, ys = map(torch.tensor, (xs, ys))
    ds = TensorDataset(xs, ys)
    if batchSize is not None:
        dl = DataLoader(ds, batch_size=batchSize, shuffle=shuffle)
    else:
        dl = DataLoader(ds, shuffle=shuffle)
    return dl


def setup(filePath, batchSize=args.batchSize, contextSize=args.contextSize, minWordCount=args.minWordCount,
          unknownToken=UNKNOWN_TOKEN, trainProportion=args.trainProportion, validProportion=args.validProportion,
          algorithm=args.algorithmType, threshold=args.subsampleThreshold,
          unigramDistributionPower=args.unigramDistributionPower):
    checkAlgorithmImplemented(algorithm)
    now = datetime.now()
    data = getData(filePath)
    wordMapping, reverseWordMapping, allowableVocab, vocabSize, frequencies, distribution = buildVocab(data,
                                                                                                       minWordCount,
                                                                                                       unknownToken,
                                                                                                       unigramDistributionPower)
    trainData, validData, testData = splitData(data, trainProportion, validProportion)
    logger.info("[" + str(datetime.now()) + "]: Train data")
    trainDl = buildDataLoader(trainData, wordMapping, frequencies, contextSize, algorithm, threshold, subSample=True,
                              batchSize=batchSize, shuffle=True)
    logger.info("[" + str(datetime.now()) + "]: Validation data")
    validDl = buildDataLoader(validData, wordMapping, frequencies, contextSize, algorithm, threshold, subSample=True,
                              batchSize=2 * batchSize, shuffle=False)
    logger.info("[" + str(datetime.now()) + "]: Test data")
    testDl = buildDataLoader(testData, wordMapping, frequencies, contextSize, algorithm, threshold, subSample=False,
                             batchSize=2 * batchSize, shuffle=False)
    seconds = (datetime.now() - now).total_seconds()
    logger.info("[" + str(datetime.now()) + "]: Setting up took: " + str(seconds) + " seconds")
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


def train(modelName, trainDl, validDl, vocabSize, distribution=None, epochs=args.epochs,
          embeddingDim=args.embeddingDimension, contextSize=args.contextSize, innerProductClamp=args.innerProductClamp,
          lr=args.learningRate, momentum=args.momentum, numNegativeSamples=args.numNegativeSamples,
          learningRateDecayFactor=args.learningRateDecayFactor, patience=args.patience, algorithm=args.algorithmType):
    checkAlgorithmImplemented(algorithm)
    logger.info("[" + str(datetime.now()) + "]: Training " + algorithm + " for " + str(epochs) +
                " epochs. Context size is " + str(contextSize) + ", embedding dimension is " + str(embeddingDim) +
                ", initial learning rate is " + str(lr) + " with a decay factor of " + str(learningRateDecayFactor) +
                " after " + str(patience) + " epochs without progress.")
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
        logger.info("[" + str(datetime.now()) + "]: Epoch: " + str(epoch))

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
                logger.info("[" + str(datetime.now()) + "]: Processed " + str(numBatchesProcessed) + " batches out of "
                            + str(len(trainDl)) + " (training)")
        trainLoss = totalLoss / len(trainDl)
        logger.info("[" + str(datetime.now()) + "]: Training loss: " + str(trainLoss))
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
                    logger.info("[" + str(datetime.now()) + "]: Processed " + str(numBatchesProcessed) +
                                " batches out of " + str(len(validDl)) + " (validation)")
        validLoss = validLoss / len(validDl)
        valLosses.append(validLoss)
        logger.info("[" + str(datetime.now()) + "]: Validation loss: " + str(validLoss))

        seconds = (datetime.now() - now).total_seconds()
        logger.info("[" + str(datetime.now()) + "]: Epoch took: " + str(seconds) + " seconds")
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


def saveModelState(model, wordMapping, reverseWordMapping, vocabulary, frequencies):
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
    logger.info("[" + str(datetime.now()) + "]: Saved model " + model.name)
    return


def loadModelState(modelName, algorithm=args.algorithmType, unigramDistributionPower=args.unigramDistributionPower):
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
    infile = open(modelName + algorithm + 'ModelData', 'rb')
    modelData = load(infile)
    infile.close()
    if algorithm.upper() == 'CBOW':
        model = ContinuousBagOfWords(len(vocabulary), modelData['embeddingDim'], modelData['contextSize'], modelName)
    elif algorithm.upper() == 'SGNS':
        model = SkipGramWithNegativeSampling(len(vocabulary), modelData['embeddingDim'], modelData['contextSize'],
                                             modelData['numNegativeSamples'], modelData['innerProductClamp'], modelName)
    model.load_state_dict(torch.load(modelName + algorithm + '.pt'))
    logger.info("[" + str(datetime.now()) + "]: Loaded model " + modelName)
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


def finalEvaluation(model, testDl, distribution=None, lossFunction=nn.NLLLoss(),
                    numNegativeSamples=args.numNegativeSamples):
    with torch.no_grad():
        testLoss = 0
        numBatchesProcessed = 0
        for xb, yb in testDl:
            if CUDA:
                xb = xb.to('cuda')
                yb = yb.to('cuda')
            if model.algorithmType == 'CBOW':
                testLoss += lossFunction(model(xb), yb).item()
            elif model.algorithmType == 'SGNS':
                negativeSamples = produceNegativeSamples(distribution, numNegativeSamples, len(yb))
                if CUDA:
                    negativeSamples = negativeSamples.to('cuda')
                loss = model(yb, xb, negativeSamples)
                testLoss += torch.mean(loss).item()
            numBatchesProcessed += 1
            if numBatchesProcessed % args.batchesForLogging == 0:
                logger.info("[" + str(datetime.now()) + "]: Processed " + str(numBatchesProcessed) + " batches out of "
                            + str(len(testDl)) + " (testing)")
        testLoss = testLoss / len(testDl)
    return testLoss


# Example usage:

wordIndex, reverseWordIndex, vocab, VOCAB_SIZE, wordFrequencies, sampleDistribution, trainDataLoader, validDataLoader, testDataLoader = setup('reviews_Grocery_and_Gourmet_Food_5.json.gz', algorithm=args.algorithmType)
trainedModel = train(args.name, trainDataLoader, validDataLoader, VOCAB_SIZE, distribution=sampleDistribution,
                     algorithm=args.algorithmType)
# print(finalEvaluation(trainedModel, testDataLoader, distribution=sampleDistribution, algorithm=algorithmType))
saveModelState(trainedModel, wordIndex, reverseWordIndex, vocab, wordFrequencies)
# wordIndex, reverseWordIndex, vocab, wordFrequencies, sampleDistribution, loadedModel = loadModelState(name,
#                                                                                                       algorithm=algorithmType)
# print(topKSimilarities(loadedModel, 'guitar', wordIndex, vocab))
# print(topKSimilaritiesAnalogy(loadedModel, 'buying', 'buy', 'sell', wordIndex, vocab))
