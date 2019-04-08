# clean implementation of a batch, check for bugs later

import spacy
from nltk import tokenize as nltk_tokenize_sentence
import pickle
import torch
import torch.nn as nn


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
		embedded = self.dropout(self.embedding(text))
		output, (hidden, cell) = self.rnn(embedded)
		hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
							
		return self.fc(hidden)


def filter(token):
	if token[0] == '@':
		return 'atakhansspecialtoken'
	if token[:4] == 'http':
		return 'httpakhansspecialtoken'
	return token.lower()


def prepare_sentence(text, target_arr):
	# returns only the first sentence where target occured
	global total_cases, counter_bad_case

	# most probably will not need this
	text = " ".join(text.split())
	text = text.replace('](https:', '] (https:')
	text = text.replace('](http:', '] (http:')
	text = text.replace('\u2019', "'")
	text = text.replace('&amp;', "&")
	# end
	arr = nltk_tokenize_sentence.sent_tokenize(text)
	target_arr = [word.lower() for word in target_arr]
	tokenized_all = []

	for item in arr:
		target_join = ' '.join(target_arr)
		target_join = ' ' + target_join + ' '
		sentence = [tok.text.lower() for tok in nlp.tokenizer(item)]
		sentence = ' '.join(sentence)
		sentence = ' ' + sentence + ' '

		if target_join in sentence:
			sentence = sentence.replace(target_join, ' my_target_wrapper' + target_join + 'my_target_wrapper ')
			sentence = sentence.strip()
			tokenized = [filter(tok.text) for tok in nlp.tokenizer(sentence)]

			tokenized_all += tokenized
			
	if len(tokenized_all) != 0:
		return tokenized_all

	# BAD CASE, SHOULDN'T HAPPEN
	tokenized = [filter(tok.text) for tok in nlp.tokenizer(arr[0])]

	return tokenized


def token2index(tokenized, stoi_dict, max_len):
	cur_len = len(tokenized)
	indexed = [stoi_dict[t] if t in stoi_dict else stoi_dict['<unk>'] for t in tokenized] + \
		+ (max_len - cur_len) * [stoi_dict['<pad>']]

	return indexed


def load_obj(name):
	with open('word-dictionaries/' + name + '.pkl', 'rb') as f:
		return pickle.load(f)


def texts2indexes(texts, target_arrs, stoi_dict):
	assert len(texts) == len(target_arrs)

	tokenized_all, indexed_all = [], []
	total = len(texts)
	max_len = 0

	for i in range(total):
		tokenized = prepare_sentence(texts[i], target_arrs[i])
		tokenized_all.append(tokenized)
		max_len = max(len(tokenized), max_len)

	for i in range(total):
		indexed = token2index(tokenized_all[i], stoi_dict, max_len)
		indexed_all.append(indexed)

	# batch_size x max_len
	return tokenized_all, indexed_all


def predict_sentiment(texts, target_arrs, model, stoi_dict, label_itos):
	assert len(texts) == len(target_arrs)
	
	tokenized_all, indexed_all = texts2indexes(texts, target_arrs, stoi_dict)

	tensor = torch.LongTensor(indexed_all).to(device)
	tensor = tensor.transpose(0, 1)

	preds = model(tensor)
	max_preds = preds.argmax(dim=1) 
	sentiments = [label_itos[max_pred_index] for max_pred_index in max_preds]
	probs = nn.functional.softmax(preds, dim=1)

	return texts, sentiments, probs, tokenized_all, indexed_all


nlp = spacy.load('en')
stoi_dict = load_obj('word-dictionary-for-model2-combined-dataset-distantLearning-model')
label_itos = load_obj('label-itos-for-model2-combined-dataset-distantLearning-model')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INPUT_DIM = 25002
EMBEDDING_DIM = 200
HIDDEN_DIM = 300 
OUTPUT_DIM = 3 
N_LAYERS = 1 
BIDIRECTIONAL = True
DROPOUT = 0.5

model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT, stoi_dict['<pad>'])
model = model.to(device)
model_save_dir = 'model2-combined-dataset-distantLearning-model.pt'

if torch.cuda.is_available():
	model.load_state_dict(torch.load(model_save_dir))
else:
	model.load_state_dict(torch.load(model_save_dir, map_location=lambda storage, loc: storage))
model.eval()



##### Andrew's doc prediction
"""
print('ANDREWS DATA')
import pickle 
import json 

with open('data-andrew-v3-pretty.json') as f:
	data = json.load(f)

texts = []
target_arrs = []
old_sentiments = []
arr_dict = []

for i, item in enumerate(data):
	texts.append(item['processed_body'])
	target_arrs.append(item['name'])
	old_sentiments.append(item['sentiment'])

	if i % 64 == 63:
		texts, sentiments, probs, tokenized_all, indexed_all = \
			predict_sentiment(texts, target_arrs, model, stoi_dict, label_itos)

		for j in range(len(texts)):
			arr_dict.append({
				'body': texts[j], 
				'target_arr': target_arrs[j],
				'old_sentiment': old_sentiments[j],
				'NEW_predicted_sentiment': sentiments[j], 
				'NEW_probs': [num.item() for num in probs[j]], 
				'NEW_tokenized': ' '.join(tokenized_all[j]), 
				'NEW_indexed': ' '.join([str(num) for num in indexed_all[j]]), 
			})
		texts = []
		target_arrs = []
		old_sentiments = []

		print(i)


with open('data-ANDREW-classified-v3.json', 'w') as outfile:
	json.dump(arr_dict, outfile)

"""
##### END


