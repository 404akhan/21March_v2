# fasttext clean implementation for dataset 1.6M tweets, two label

import torch
from torchtext import data
from torchtext import datasets
import random

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import spacy


def generate_bigrams(x):
    n_grams = set(zip(*[x[i:] for i in range(2)]))
    for n_gram in n_grams:
        x.append(' '.join(n_gram))
    return x


class FastText(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim, padding_idx):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.fc = nn.Linear(embedding_dim, output_dim)
        
    def forward(self, text):
        
        #text = [sent len, batch size]
        
        embedded = self.embedding(text)
                
        #embedded = [sent len, batch size, emb dim]
        
        embedded = embedded.permute(1, 0, 2)
        
        #embedded = [batch size, sent len, emb dim]
        
        pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1) 
        
        #pooled = [batch size, embedding_dim]
                
        return self.fc(pooled)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def categorical_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_preds = preds.argmax(dim=1, keepdim=True) # get the index of the max probability
    correct = max_preds.squeeze(1).eq(y)
    return correct.sum()/torch.FloatTensor([y.shape[0]])

def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        
        optimizer.zero_grad()
        
        predictions = model(batch.text).squeeze(1)
        
        loss = criterion(predictions, batch.label)
        
        acc = categorical_accuracy(predictions, batch.label)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:

            predictions = model(batch.text)
            
            loss = criterion(predictions, batch.label)
            
            acc = categorical_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def filter(token):
	if token[0] == '@':
		return '<at_@>'
	if token[:4] == 'http':
		return '<http>'
	return token.lower()


def predict_sentiment(sentence, min_len=4):
    nlp = spacy.load('en')
    tokenized = generate_bigrams([filter(tok.text) for tok in nlp.tokenizer(sentence)])
    if len(tokenized) < min_len:
        tokenized += ['<pad>'] * (min_len - len(tokenized))
    indexed = [POST.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    preds = model(tensor)
    max_preds = preds.argmax(dim=1)
    sentiment = LABEL.vocab.itos[max_preds.item()]
    return sentence, sentiment


print(generate_bigrams(['This', 'film', 'is', 'terrible']))

SEED = 1234

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

POST = data.Field(preprocessing=generate_bigrams)
LABEL = data.LabelField()

fields = {'post': ('text', POST), 'sentiment': ('label', LABEL)}
train_path = '../dataset_sentiment_not_target_semeval/dataset_sentiment_not_target/merged_dataset.json'
test_fake_path = 'fake.json'

train_data, _ = data.TabularDataset.splits(
                            path = 'my_data_v2',
                            train = train_path,
                            test = test_fake_path,
                            format = 'json',
                            fields = fields
)
train_data, test_data = train_data.split(random_state=random.seed(SEED))
train_data, valid_data = train_data.split(random_state=random.seed(SEED))

POST.build_vocab(train_data, max_size=25000, vectors="glove.twitter.27B.100d", unk_init=torch.Tensor.normal_)
LABEL.build_vocab(train_data)

BATCH_SIZE = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data), sort_key=lambda x: len(x.text),
    batch_size=BATCH_SIZE,
    device=device)

INPUT_DIM = len(POST.vocab)
EMBEDDING_DIM = 100
OUTPUT_DIM = 3

model = FastText(INPUT_DIM, EMBEDDING_DIM, OUTPUT_DIM, POST.vocab.stoi['<pad>'])
print('The model has %d trainable parameters' % count_parameters(model))

pretrained_embeddings = POST.vocab.vectors
model.embedding.weight.data.copy_(pretrained_embeddings)
model.embedding.weight.data[POST.vocab.stoi['<unk>']] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[POST.vocab.stoi['<pad>']] = torch.zeros(EMBEDDING_DIM)

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

model = model.to(device)
criterion = criterion.to(device)

N_EPOCHS = 10
best_valid_loss = float('inf')

print('start')
print(vars(train_data.examples[0]))

print("Unique tokens in POST vocabulary: %d" % len(POST.vocab))
print("Unique tokens in LABEL vocabulary: %d" % len(LABEL.vocab))

print(POST.vocab.freqs.most_common(20))
print(POST.vocab.itos[:10])
print(LABEL.vocab.stoi)

print(len(train_data), len(valid_data), len(test_data))


for epoch in range(N_EPOCHS):

    start_time = time.time()
    
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
    
    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'model1-model.pt')
    
    print('Epoch: %d | Epoch Time: %dm %ds' % (epoch+1, epoch_mins, epoch_secs))
    print('\tTrain Loss: %.3f | Train Acc: %.2f%%' % (train_loss, train_acc*100))
    print('\t Val. Loss: %.3f |  Val. Acc: %.2f%%' % (valid_loss, valid_acc*100))


model.load_state_dict(torch.load('model1-model.pt'))
test_loss, test_acc = evaluate(model, test_iterator, criterion)

print('Test Loss: %.3f | Test Acc: %.2f%%' % (test_loss, test_acc*100))

print(predict_sentiment("This film is terrible"))
print(predict_sentiment("This film is great"))
