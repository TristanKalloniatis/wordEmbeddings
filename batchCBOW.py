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

CONTEXT_SIZE = 5
EMBEDDING_DIM = 16
MIN_WORD_COUNT = 10
TRAIN_PROPORTION = 0.75
VALID_PROPORTION = 0.15
LEARNING_RATE = 0.5
MOMENTUM = 0.9
BATCH_SIZE = 100
EPOCHS = 25
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
    vocabularySize = len(allowableVocab)
    wordMapping = {word: i for i, word in enumerate(allowableVocab)}
    reverseWordMapping = {i: word for word, i in wordMapping.items()}
    reverseWordMapping[len(allowableVocab)] = '???'
    for word in wordCounts:
        if wordCounts[word] < MIN_WORD_COUNT:
            wordMapping[word] = len(allowableVocab)
    if rareWordsExist:
        vocabularySize += 1
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
                                                                int(len(rawData) * (TRAIN_PROPORTION + VALID_PROPORTION)):]


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


data = getData('Downloads/reviews_Musical_Instruments_5.json.gz')
wordIndex, reverseWordIndex, vocab, VOCAB_SIZE = buildVocab(data)
trainData, validData, testData = splitData(data)

print("Train data")
trainDl = buildDataLoader(trainData, wordIndex, batchSize=BATCH_SIZE, shuffle=True)
print("Validation data")
validDl = buildDataLoader(validData, wordIndex, batchSize=2 * BATCH_SIZE)
print("Test data")
testDl = buildDataLoader(testData, wordIndex, batchSize=2 * BATCH_SIZE)


class continuousBagOfWords(nn.Module):
    def __init__(self, vocabSize, embeddingDim, contextSize):
        super().__init__()
        self.embeddings = nn.Embedding(vocabSize, embeddingDim)
        self.linear = nn.Linear(embeddingDim, vocabSize)
        self.contextSize = contextSize

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        out = self.linear(embeds.sum(dim=-2))
        logProbs = F.log_softmax(out, dim=-1)
        return logProbs


trainLosses = []
valLosses = []
lossFunction = nn.NLLLoss()
model = continuousBagOfWords(VOCAB_SIZE, EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, nesterov=True)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=LEARNING_RATE_DECAY_FACTOR, patience=PATIENCE, verbose=True)

for epoch in range(EPOCHS):
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

with torch.no_grad():
    testLoss = sum(lossFunction(model(xb), yb).item() for xb, yb in testDl)
    test_loss = testLoss / len(testDl)
    print(testLoss)
