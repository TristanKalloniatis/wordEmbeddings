import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

CONTEXT_SIZE = 2
EMBEDDING_DIM = 10
raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()
vocab = set(raw_text)
VOCAB_SIZE = len(vocab)
wordIndex = {word: i for i, word in enumerate(vocab)}
data = []
for i in range(CONTEXT_SIZE, len(raw_text) - CONTEXT_SIZE):
    context = [wordIndex[raw_text[i - j - 1]] for j in range(CONTEXT_SIZE)]
    context += [wordIndex[raw_text[i + j + 1]] for j in range(CONTEXT_SIZE)]
    target = wordIndex[raw_text[i]]
    data.append((context, target))


class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
        self.context_size = context_size

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view(2 * self.context_size, -1)
        out = self.linear(embeds.sum(dim=0).view(1, -1))
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


losses = []
loss_function = nn.NLLLoss()
model = CBOW(VOCAB_SIZE, EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.001)

for epoch in range(10):  # This training is not currently sensible
    total_loss = 0
    for context, target in data:
        model.zero_grad()
        log_probs = model(torch.tensor(context, dtype=torch.long))
        loss = loss_function(log_probs, torch.tensor([target], dtype=torch.long))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    losses.append(total_loss)
print(losses)

testWord = 'manipulate'
print(testWord)
print(model.embeddings(torch.tensor([wordIndex[testWord]], dtype=torch.long)))
