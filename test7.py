# create datasets

import csv
import spacy
import json
import numpy as np


nlp = spacy.load('en')


fname = 'my_data_v2/training.1600000.processed.noemoticon.csv'
f = open(fname, 'r', errors='ignore')
reader = csv.reader(f)

arr = []
for row in reader:
	arr.append(row)


def filter(token):
	if token[0] == '@':
		return '<at_@>'
	if token[:4] == 'http':
		return '<http>'
	return token.lower()


def label_change(label):
	if label == "4":
		return 'positive'
	if label == "2":
		return 'neutral'
	if label == "0":
		return 'negative'
	return 'ERROR_NO_LABEL'


def main(fname_out='my_data_v2/data.json'):
	counter_dict = {'positive': 0, 'neutral': 0, 'negative': 0, 'ERROR_NO_LABEL': 0}
	open(fname_out, 'w').close()

	with open(fname_out, 'w') as outfile:  
		row_count = len(arr)
		indexes = np.random.permutation(row_count)
		for counter, i in enumerate(indexes):
			row = arr[i]

			tokenized = [filter(tok.text) for tok in nlp.tokenizer(row[-1])]
			label = label_change(row[0])
			counter_dict[label] += 1

			data = {'post': tokenized, 'label': label}
			json.dump(data, outfile)
			outfile.write("\n")
	print(counter_dict)


# main()

f.close()

