# rnn clean implementation for dataset 1.6M tweets, two label

import torch
from torchtext import data
from torchtext import datasets
import random
import torch.nn as nn
import torch.optim as optim
import time
import spacy
from sklearn.metrics import f1_score
import numpy as np 


class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, 
                 output_dim, n_layers, bidirectional, dropout, padding_idx):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, 
                           bidirectional=bidirectional, dropout=dropout)
        self.fc = nn.Linear(hidden_dim*2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        
        #text = [sent len, batch size]
        
        embedded = self.dropout(self.embedding(text))
        
        #embedded = [sent len, batch size, emb dim]
        
        output, (hidden, cell) = self.rnn(embedded)
        
        #output = [sent len, batch size, hid dim * num directions]
        #hidden = [num layers * num directions, batch size, hid dim]
        #cell = [num layers * num directions, batch size, hid dim]
        
        #concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        #and apply dropout
        
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
                
        #hidden = [batch size, hid dim * num directions]
            
        return self.fc(hidden)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def categorical_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_preds = preds.argmax(dim=1, keepdim=True) # get the index of the max probability
    correct = max_preds.squeeze(1).eq(y)
    return correct.sum()/torch.FloatTensor([y.shape[0]])


def f1_scores(preds, y):
    f1_macro = f1_score(preds, y, average='macro')

    f1_weighted = f1_score(preds, y, average='weighted')

    return f1_macro, f1_weighted


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
    predictions_arr = []
    labels_arr = []

    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:

            predictions = model(batch.text).squeeze(1)
            
            loss = criterion(predictions, batch.label)

            # f1_score code
            predictions_arr += predictions.argmax(dim=1).tolist()
            labels_arr += batch.label.tolist()
            # f1_score end
            
            acc = categorical_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    f1_macro, f1_weighted = f1_scores(predictions_arr, labels_arr) 
    return epoch_loss / len(iterator), epoch_acc / len(iterator), f1_macro, f1_weighted



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


def predict_sentiment(sentence):
    nlp = spacy.load('en')
    tokenized = [filter(tok.text) for tok in nlp.tokenizer(sentence)]
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    preds = model(tensor)
    max_preds = preds.argmax(dim=1) 
    sentiment = LABEL.vocab.itos[max_preds.item()]

    probs = nn.functional.softmax(preds, dim=1)[0].tolist()
    return sentence, sentiment, probs


SEED = 1234

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

TEXT = data.Field()
LABEL = data.LabelField()

fields = {'post': ('text', TEXT), 'label': ('label', LABEL)}

train_data, _ = data.TabularDataset.splits(
                            path = 'my_data_v2',
                            train = 'data.json',
                            test= 'fake.json',
                            format = 'json',
                            fields = fields
)

train_data, test_data = train_data.split(random_state=random.seed(SEED))
train_data, valid_data = train_data.split(random_state=random.seed(SEED))

TEXT.build_vocab(train_data, vectors="glove.twitter.27B.200d", unk_init=torch.Tensor.normal_)
LABEL.build_vocab(train_data)

BATCH_SIZE = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data), sort_key=lambda x: len(x.text),
    batch_size=BATCH_SIZE,
    device=device)

INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 200
HIDDEN_DIM = 300 # !!! change -> 256
OUTPUT_DIM = 2 # !!! change -> 3
N_LAYERS = 1 # !!! change -> 2
BIDIRECTIONAL = True
DROPOUT = 0.5

model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT, TEXT.vocab.stoi['<pad>'])

pretrained_embeddings = TEXT.vocab.vectors
print(pretrained_embeddings.shape)
model.embedding.weight.data.copy_(pretrained_embeddings)
model.embedding.weight.data[TEXT.vocab.stoi['<unk>']] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[TEXT.vocab.stoi['<pad>']] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.requires_grad = False

print('The model has %d trainable parameters' % count_parameters(model))

optimizer = optim.Adam([param for param in model.parameters() if param.requires_grad == True], weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()

model = model.to(device)
criterion = criterion.to(device)

N_EPOCHS = 25
best_valid_loss = float('inf')

model_save_dir = 'model2-save-1-6M-embeddings-model.pt'

print('start')
print(vars(train_data.examples[0]))

print("Unique tokens in TEXT vocabulary: %d" % len(TEXT.vocab))
print("Unique tokens in LABEL vocabulary: %d" % len(LABEL.vocab))

print(TEXT.vocab.freqs.most_common(20))
print(TEXT.vocab.itos[:10])
print(LABEL.vocab.freqs.most_common(20))
print(LABEL.vocab.stoi)

print("train_data %d, valid_data %d, test_data %d" % 
    (len(train_data), len(valid_data), len(test_data)))


for epoch in range(N_EPOCHS):

    start_time = time.time()
    
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc, f1_macro, f1_weighted = evaluate(model, valid_iterator, criterion)
    
    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), model_save_dir)

    if epoch == 10:
        print('Unfreeze embeddings')
        model.embedding.weight.requires_grad = True
        optimizer = optim.Adam(model.parameters())

    
    print('Epoch: %d | Epoch Time: %dm %ds' % (epoch+1, epoch_mins, epoch_secs))
    print('\tTrain Loss: %.3f | Train Acc: %.2f%%' % (train_loss, train_acc*100))
    print('\t Val. Loss: %.3f |  Val. Acc: %.2f%% | F1_macro: %.3f | F1_weighted: %.3f' % (valid_loss, valid_acc*100, f1_macro, f1_weighted))

import os
if os.path.exists(model_save_dir):
    model.load_state_dict(torch.load(model_save_dir))

test_loss, test_acc, f1_macro, f1_weighted = evaluate(model, test_iterator, criterion)
print('Test Loss: %.3f | Test Acc: %.2f%% | F1_macro: %.3f | F1_weighted: %.3f' % 
    (test_loss, test_acc*100, f1_macro, f1_weighted))

print(predict_sentiment("This film is terrible"))
print(predict_sentiment("This film is great"))



from tqdm import tqdm

def write_embeddings(path, embeddings, vocab):
    print('writing embeddings into: %s' % path)
    
    with open(path, 'w') as f:
        for i, embedding in enumerate(tqdm(embeddings)):
            word = vocab.itos[i]
            #skip words with unicode symbols
            if len(word) != len(word.encode()):
                continue
            vector = ' '.join([str(i) for i in embedding.tolist()])
            f.write('%s %s\n' % (word, vector))


write_embeddings('.vector_cache/1-6M-my-train-embedding-200d-v2.txt', 
                 model.embedding.weight.data, 
                 TEXT.vocab)



# print(len(TEXT.vocab.stoi))
# print(len(TEXT.vocab.itos))
# print(TEXT.vocab.stoi)
# print(TEXT.vocab.itos)
# print(TEXT.vocab.freqs.most_common(40))

# print(len(TEXT.vocab.vectors))
# print(TEXT.vocab.vectors[:10])
# print(TEXT.vocab.itos[:10])
# print(len(model.embedding.weight.data))

# with open(fname_out, 'w') as outfile:  
#     whole_str = ''

#     for i in range(len(TEXT.vocab.itos)):
#         token = TEXT.vocab.itos[i]

#         weights = model.embedding.weight.data[TEXT.vocab.stoi[token]]
#         str_weights = token + ' '
#         for w in weights:
#             str_weights +=  ('%.5f ' % w.item())
#         str_weights += '\n'

#         whole_str += str_weights

#     outfile.write(whole_str)


# 25000 change to no-limit
# small_data to data.json