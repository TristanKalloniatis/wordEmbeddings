import gzip
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import pickle

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


def getData(filePath):
    file = gzip.open(filePath, mode='rb')
    rawData = []
    for line in file:
        rawData.append(json.loads(line))
    file.close()
    print("Number of reviews:", len(rawData))
    return rawData


def preProcess(text):
    allowableChars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
                      't', 'u', 'v', 'w', 'x', 'y', 'z', ' ']
    text = text.lower()
    result = ''
    for x in text:
        if x in allowableChars:
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


def buildVocab(rawData):
    wordCounts = buildWordCounts(rawData)
    allowableVocab = []
    rareWordsExist = False
    for word in wordCounts:
        if wordCounts[word] >= MIN_WORD_COUNT:
            allowableVocab.append(word)
        else:
            rareWordsExist = True
    wordMapping = {word: i for i, word in enumerate(allowableVocab)}
    reverseWordMapping = {i: word for word, i in wordMapping.items()}
    if rareWordsExist:
        reverseWordMapping[len(allowableVocab)] = '???'
    for word in wordCounts:
        if wordCounts[word] < MIN_WORD_COUNT:
            wordMapping[word] = len(allowableVocab)
    if rareWordsExist:
        allowableVocab.append('???')
    vocabularySize = len(allowableVocab)
    print("Vocabulary size:", vocabularySize)
    return wordMapping, reverseWordMapping, allowableVocab, vocabularySize


def preProcessWords(words, wordMapping):
    dataPoints = []
    for i in range(CONTEXT_SIZE, len(words) - CONTEXT_SIZE):
        context = [wordMapping[words[i - j - 1]] for j in range(CONTEXT_SIZE)]
        context += [wordMapping[words[i + j + 1]] for j in range(CONTEXT_SIZE)]
        target = wordMapping[words[i]]
        dataPoints.append((context, target))
    return dataPoints


def splitData(rawData):
    return rawData[: int(len(rawData) * TRAIN_PROPORTION)], rawData[int(len(rawData) * TRAIN_PROPORTION): int(
        len(rawData) * (TRAIN_PROPORTION + VALID_PROPORTION))], rawData[
                                                                int(len(rawData) * (
                                                                        TRAIN_PROPORTION + VALID_PROPORTION)):]


def buildDataLoader(rawData, wordMapping, batchSize=None, shuffle=False):
    xs = []
    ys = []
    for review in rawData:
        dataPoints = preProcessWords(preProcess(review['reviewText']).split(), wordMapping)
        for dataPointX, dataPointY in dataPoints:
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


def setup(filePath):
    data = getData(filePath)
    wordMapping, reverseWordMapping, allowableVocab, vocabularySize = buildVocab(data)
    trainData, validData, testData = splitData(data)
    print("Train data")
    trainDl = buildDataLoader(trainData, wordMapping, batchSize=BATCH_SIZE, shuffle=True)
    print("Validation data")
    validDl = buildDataLoader(validData, wordMapping, batchSize=2 * BATCH_SIZE)
    print("Test data")
    testDl = buildDataLoader(testData, wordMapping, batchSize=2 * BATCH_SIZE)
    return wordMapping, reverseWordMapping, allowableVocab, vocabularySize, trainDl, validDl, testDl


class continuousBagOfWords(nn.Module):
    def __init__(self, vocabSize, embeddingDim, contextSize):
        super().__init__()
        self.embeddings = nn.Embedding(vocabSize, embeddingDim)
        self.linear = nn.Linear(embeddingDim, vocabSize)
        self.contextSize = contextSize
        self.embeddingDim = embeddingDim

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        out = self.linear(embeds.sum(dim=-2))
        logProbs = F.log_softmax(out, dim=-1)
        return logProbs


def train(trainDl, validDl, vocabSize, epochs=EPOCHS, embeddingDim=EMBEDDING_DIM, contextSize=CONTEXT_SIZE,
          lr=LEARNING_RATE, momentum=MOMENTUM, learningRateDecayFactor=LEARNING_RATE_DECAY_FACTOR, patience=PATIENCE):
    trainLosses = []
    valLosses = []
    lossFunction = nn.NLLLoss()
    model = continuousBagOfWords(vocabSize, embeddingDim, contextSize)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=True)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=learningRateDecayFactor, patience=patience,
                                  verbose=True)

    for epoch in range(epochs):
        print("Epoch:", epoch)
        now = datetime.now()

        model.train()
        totalLoss = 0
        for xb, yb in trainDl:
            predictions = model(xb)
            loss = lossFunction(predictions, yb)
            loss.backward()
            totalLoss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
        trainLoss = totalLoss / len(trainDl)
        print("Training loss:", trainLoss)
        trainLosses.append(trainLoss)

        model.eval()
        with torch.no_grad():
            validLoss = sum(lossFunction(model(xb), yb) for xb, yb in validDl).item()
        validLoss = validLoss / len(validDl)
        valLosses.append(validLoss)
        print("Validation loss:", validLoss)

        seconds = (datetime.now() - now).total_seconds()
        print("Took:", seconds)
        scheduler.step(validLoss)

    plt.plot(trainLosses, 'go--', linewidth=2, markersize=12)
    plt.plot(valLosses, 'bo--', linewidth=2, markersize=12)

    return model


def saveModelState(model, modelName, wordMapping, reverseWordMapping, vocabulary):
    torch.save(model.state_dict(), modelName + '.pt')
    outfile = open(modelName + 'WordMapping', 'wb')
    pickle.dump(wordMapping, outfile)
    outfile.close()
    outfile = open(modelName + 'reverseWordMapping', 'wb')
    pickle.dump(reverseWordMapping, outfile)
    outfile.close()
    outfile = open(modelName + 'Vocab', 'wb')
    pickle.dump(vocabulary, outfile)
    outfile.close()
    modelData = {'embeddingDim': model.embeddingDim, 'contextSize': model.contextSize}
    outfile = open(modelName + 'ModelData', 'wb')
    pickle.dump(modelData, outfile)
    outfile.close()


def loadModelState(modelName):
    infile = open(modelName + 'wordMapping', 'rb')
    wordMapping = pickle.load(infile)
    infile.close()
    infile = open(modelName + 'reverseWordMapping', 'rb')
    reverseWordMapping = pickle.load(infile)
    infile.close()
    infile = open(modelName + 'Vocab', 'rb')
    vocab = pickle.load(infile)
    infile.close()
    infile = open(modelName + 'ModelData', 'rb')
    modelData = pickle.load(infile)
    infile.close()
    model = continuousBagOfWords(len(vocab), modelData['embeddingDim'], modelData['contextSize'])
    model.load_state_dict(torch.load(modelName + '.pt'))
    model.eval()
    return wordMapping, reverseWordMapping, vocab, model


def topKSimilarities(model, word, wordMapping, vocabulary, K=10):
    allSimilarities = {}
    with torch.no_grad():
        wordEmbedding = model.embeddings(torch.tensor(wordMapping[word], dtype=torch.long))
        for otherWord in vocabulary:
            otherEmbedding = model.embeddings(torch.tensor(wordMapping[otherWord], dtype=torch.long))
            allSimilarities[otherWord] = nn.CosineSimilarity(dim=0)(wordEmbedding, otherEmbedding).item()
    return {k: v for k, v in sorted(allSimilarities.items(), key=lambda item: item[1], reverse=True)[1:K+1]}


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


def finalEvaluation(model, testDl, lossFunction=nn.NLLLoss()):
    with torch.no_grad():
        testLoss = sum(lossFunction(model(xb), yb).item() for xb, yb in testDl)
        testLoss = testLoss / len(testDl)
    return testLoss

# Example usage:

# wordIndex, reverseWordIndex, vocab, VOCAB_SIZE, trainDl, validDl, testDl = setup('Downloads/reviews_Grocery_and_Gourmet_Food_5.json.gz')
# trainedModel = train(trainDl, validDl, vocabSize)
# print(finalEvaluation(trainedModel, testDl))
# saveModelState(trainedModel, 'CBOW', wordIndex, reverseWordIndex, vocab)
# wordIndex, reverseWordIndex, vocab, loadedModel = loadModelState('CBOW')
# print(topKSimilarities(loadedModel, 'apple', wordIndex, vocab))
# print(topKSimilaritiesAnalogy(loadedModel, 'buying', 'buy', 'sell', wordIndex, vocab))
