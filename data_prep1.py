# 1. prepare data for sent-anal using semeval
# 2. prepare data for target sent-anal using semeval + old twitter

# 3. (now) try old twitter only

import spacy
import json

nlp = spacy.load('en')


def get_label(c):
	if c == '-1':
		return 'negative'
	elif c == '1':
		return 'positive'
	elif c == '0':
		return 'neutral'
	else:
		return 'ERROR_SHOULDNT_REACH'


def filter(token):
	if token[0] == '@':
		return '<at_@>'
	if token[:4] == 'http':
		return '<http>'
	return token.lower()


def load_and_save(fname, fname_out):
	with open(fname) as f:
	    content = f.readlines()
	# you may also want to remove whitespace characters like `\n` at the end of each line
	content = [x.strip() for x in content] 

	with open(fname_out, 'w') as outfile:  
		for i in range(0, len(content), 3):
			post = content[i]
			target = content[i+1]
			label = get_label(content[i+2])

			post = post.replace("$T$", "my_target_wrapper " + target + " my_target_wrapper")
			tokenized = [filter(tok.text) for tok in nlp.tokenizer(post)]

			data = {'post': tokenized, 'label': label}
			json.dump(data, outfile)
			outfile.write("\n")


load_and_save('my_data/train.raw', 'my_data_v3/old_twitter_train.json')
load_and_save('my_data/test.raw', 'my_data_v3/old_twitter_test.json')
