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


def filter(token):
    if token[0] == '@':
        return '<at_@>'
    if token[:4] == 'http':
        return '<http>'
    return token.lower()


def prepare_sentence(text, target_arr):
	# returns only the first sentence where target occured

	text = text.replace('\n', ' ')
	arr = nltk_tokenize_sentence.sent_tokenize(text)
	target_arr = [word.lower() for word in target_arr]

	for item in arr:

		target_join = ' '.join(target_arr)
		target_join = ' ' + target_join + ' '
		tokenized = [filter(tok.text) for tok in nlp.tokenizer(item)]
		sentence = ' '.join(tokenized)

		if target_join in sentence:
			sentence = sentence.replace(target_join, ' my_target_wrapper' + target_join + 'my_target_wrapper ')
			tokenized = [filter(tok.text) for tok in nlp.tokenizer(sentence)]

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
	print(len(indexed_all), len(indexed_all[0]))

	tensor = torch.LongTensor(indexed_all).to(device)
	tensor = tensor.transpose(0, 1)

	print(tensor.shape)
	preds = model(tensor)

	print(preds.shape)
	max_preds = preds.argmax(dim=1) 

	print(max_preds.shape)

	sentiments = [label_itos[max_pred_index] for max_pred_index in max_preds]
	print('sentiments', sentiments)

	probs = nn.functional.softmax(preds, dim=1)

	print('probs', probs.shape)
	print(probs)
	return texts, sentiments, probs, tokenized_all, indexed_all


nlp = spacy.load('en')
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
model.eval()


texts = ['i love you barack obama!', 'adding more words, i hate you barack obama!']
target_arrs = [['barack', 'obama'], ['barack', 'obama']]


print(predict_sentiment(texts, target_arrs, model, stoi_dict, label_itos))



""" Simple test
text = "Hi everyone,\n\n&amp;#x200B;\n\nHere's my situation:\n\n \n\n\\-Introduce yourself\n\nI'm a software developer who has just relocated to Seattle, WA from Australia from January this year (2019)\n\n&amp;#x200B;\n\n\\-Age / Industry / Location\n\n32, Software Developer, Seattle WA.\n\n&amp;#x200B;\n\n\\-General goals\n\nHave the option of retiring as early as possible. I will still likely work even after I FIRE but there's nothing like having the option to stop when I want.\n\n&amp;#x200B;\n\n\\-Target FIRE Age / Amount / Withdrawal Rate / Location\n\nNo target age yet.\n\nLet's say about $30,000 a year but I don't know what my expenses fully are yet.\n\nMay retire back in Australia, or in Japan depending on circumstances. If I end up staying in the U.S, then here probably.\n\n&amp;#x200B;\n\n\\-Educational background and plans\n\nB.A in Arts (Visual Communication). Has almost nothing to do with what I do for work.\n\nNo plans at the moment.\n\n&amp;#x200B;\n\n\\-Career situation and plans\n\nJust hired at my new company but it should be stable.\n\n&amp;#x200B;\n\n\\-Current and future income breakdown, including one-time events\n\nBase of $140,000, bonus of $60,000 this year with and $40,000 next year.\n\n&amp;#x200B;\n\n\\-Budget breakdown\n\nRent: $1700\\~ per month with possibly 1% increase per year.\n\nGroceries/Day-to-day expenses: at most $2000 a month I would suspect though it will probably be closer to about $1500 really. This first month in the U.S showed up as $900\\~ in grocery expenses, and a few grand in furnishing my wife and I's small apartment which is now complete.\n\nUtilities:\n\n\\- Water &amp; trash : $70 a month so far\n\n\\- electricity: Haven't received our first bill yet\n\n\\- internet: Free for the first year but about $550 per year there-after.\n\n\\- cell phones: We own our cell phones outright, no plans to buy new ones. We pay $50 per month with h2o wireless for 2 lines.\n\n&amp;#x200B;\n\n\\-Asset breakdown, including home, cars, etc.\n\nOwn an apartment in Melbourne, Australia that we bought for $345,000 AUD, currently mortgaged at about $260,000 AUD remaining.\n\nNo cars, no plans for cars either since we live 10 minutes walk away from work.\n\n&amp;#x200B;\n\nMe:\n\n\\- $50,000 AUD in superannuation (401(k) equiv)\n\n\\- $8000 AUD invested in Vanguard ASX200 etf.\n\n\\- $5000 AUD in cash in my Australian bank account.\n\n\\- $16,000 USD in US bank account.\n\nWife:\n\n\\- $10,000 AUD in superannuation\n\n\\- $6,000 AUD invested in Vanguard ASX200 etf.\n\n\\- $20,000 AUD invested in Vanguard Australian Bond etf.\n\n\\- $45,000 AUD in a 2.8% interest rate savings account in Australian bank.\n\n&amp;#x200B;\n\nTogether:\n\n\\- $360 per week in rental income from our Melbourne property.\n\n&amp;#x200B;\n\n\\-Debt breakdown\n\nNo Australian credit cards or personal loans.\n\nThe above mentioned mortgage at $1550 AUD per month with about 4.39% interest rate.\n\nWe've opened up a Bank of America credit card under my name that has a $10,000 limit. Currently have a balance of $1500. We are using this credit card to build credit history.\n\n&amp;#x200B;\n\n\\-Health concerns\n\nNothing at the moment.\n\n&amp;#x200B;\n\n\\-Family: current situation / future plans / special needs / elderly parents\n\nMarried to my wife, she is 7 years older than me. No special needs other than dental work that my wife needs done and she has already started dealing with, with the help of my dental insurance.\n\n&amp;#x200B;\n\n\\-Other info\n\nI've just opened up a TD-Ameritrade account with the notion of buying S&amp;P500 etfs\n\n&amp;#x200B;\n\n\\-Questions?\n\nWhat can I do to improve my situation?"
target_arr = ['AUD', 'invested']

text = "Hey man I know the feeling. My brother and I have our birthdays exactly a week apart with his coming first. Well he celebrates on his day with help of parents and when my birthday come up we have a shared party that really is mostly for him. I\u2019m the older brother so it doesn\u2019t bother that much since he\u2019s happy but to be honest it kinda sucks sometimes. \n\n.\n\nMy birthday was This last Sunday and I have been sick for a while but better enough to hang out. Well mom said I couldn\u2019t join them cuz I was sick, okay understandable because I don\u2019t wanna pass it along, and after they did there thing they really didn\u2019t offer anything for me to join up for. So they rented so fancy butterfly door bmw and had gone to casinos and such. The day of my actually bday mom flew to Florida for work and brother flew back to school and I just sat at home with gf watching movies. \n\n.\n\nMy dad didn\u2019t even call me till today and said he just remembered it was my bday and wished happy belated. I was surprised since we literally hung out the Friday so I thought he would say something then.\n\n.\n\nAnyways that was a lot and I just wanted to share that I know it\u2019s hard man but hit me up if you wanna share more. "
target_arr = ['to', 'florida']

tokenized = prepare_sentence(text, target_arr)
indexed = token2index(tokenized, stoi_dict, 100)

print(tokenized)
print(indexed)
"""