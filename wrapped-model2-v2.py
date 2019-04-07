# rnn wrapped model to predict

import torch
from torchtext import data
from torchtext import datasets
import torchtext.vocab as vocab
import random
import torch.nn as nn
import torch.optim as optim
import time
import spacy
from sklearn.metrics import f1_score
import numpy as np 
import pickle
nlp = spacy.load('en')


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


def prepare_sentence(sentence, target, max_len):
    

def predict_sentiment(sentence, model, stoi_dict, label_itos):
    # TODO 1: make batch
    # TODO 2: if text has several sentences, count those who contain word, average over those or choose first one (depend on data)

    tokenized = [filter(tok.text) for tok in nlp.tokenizer(sentence)]
    indexed = [stoi_dict[t] for t in tokenized]
    print(tokenized)
    print(indexed)
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    preds = model(tensor)
    max_preds = preds.argmax(dim=1) 
    sentiment = label_itos[max_preds.item()]

    probs = nn.functional.softmax(preds, dim=1)[0].tolist()
    return sentence, sentiment, probs


def load_obj(name ):
    with open('word-dictionaries/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


stoi_dict = load_obj('word-dictionary-for-model2-combined-dataset-distantLearning-model')
label_itos = load_obj('label-itos-for-model2-combined-dataset-distantLearning-model')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INPUT_DIM = 25002
EMBEDDING_DIM = 200
HIDDEN_DIM = 300 # !!! change -> 256
OUTPUT_DIM = 3 # !!! change -> 3
N_LAYERS = 1 # !!! change -> 2
BIDIRECTIONAL = True
DROPOUT = 0.5

model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT, stoi_dict['<pad>'])
model = model.to(device)
model_save_dir = 'model2-combined-dataset-distantLearning-model.pt'


if torch.cuda.is_available():
    model.load_state_dict(torch.load(model_save_dir))
else:
    model.load_state_dict(torch.load(model_save_dir, map_location=lambda storage, loc: storage))

# test_loss, test_acc, f1_macro, f1_weighted = evaluate(model, test_target_iterator, criterion)
# print('TARGET size(%d): Test Loss: %.3f | Test Acc: %.2f%% | F1_macro: %.3f | F1_weighted: %.3f' % 
#     (len(test_target_data), test_loss, test_acc*100, f1_macro, f1_weighted))

model.eval()

print(predict_sentiment("This film is terrible", model, stoi_dict, label_itos))
print('\n')
print(predict_sentiment("my_target_wrapper This film my_target_wrapper is terrible", model, stoi_dict, label_itos))
print('\n')
print(predict_sentiment('my_target_wrapper Trump my_target_wrapper is bad!!', model, stoi_dict, label_itos))
print('\n')


##### Andrew's doc prediction
"""
model.eval()

import pickle 
import json 

with open('data-ANDREW.json') as f:
    data = json.load(f)

new_arr = []
texts = []

for i, item in enumerate(data):
    text = item['body'].lower()
    target = ' '.join(item['name'])
    text = text.replace(target, 'my_target_wrapper ' + target + ' my_target_wrapper')
    texts.append(text)

    if i >= 10:
        break

out_sentences, predicted_sentiment, probabilities, transfered_sentence, indexes_available = \
    predict_sentiment(texts, model, stoi_dict, label_itos)

print(out_sentences, predicted_sentiment, probabilities, transfered_sentence, indexes_available )

exit(0)

new_dict = dict(item)
new_dict['NEW_predicted_sentiment'] = predicted_sentiment
new_dict['NEW_probabilities'] = probabilities
new_dict['NEW_transfered_sentence'] = transfered_sentence
new_dict['NEW_indexes_available'] = indexes_available

new_arr.append(new_dict)


with open('data-ANDREW-classified.json', 'w') as outfile:
    json.dump(new_arr, outfile)
"""
##### END