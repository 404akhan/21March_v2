# merge datasets first
import json
import spacy

nlp = spacy.load('en')


def func1():
	files = [
		'twitter-2013dev-A.txt',
		'twitter-2013test-A.txt',
		'twitter-2013train-A.txt',
		'twitter-2014sarcasm-A.txt',
		'twitter-2014test-A.txt',
		'twitter-2015test-A.txt',
		'twitter-2015train-A.txt',
		'twitter-2016dev-A.txt',
		'twitter-2016devtest-A.txt',
		'twitter-2016test-A.txt',
		'twitter-2016train-A.txt',
	]

	prefix = 'dataset_sentiment_not_target_semeval/dataset_sentiment_not_target/'

	fname_out = prefix + 'merged_dataset.json'

	with open(fname_out, 'w') as outfile:  
		for filename in files:
			filepath = prefix + filename

			with open(filepath) as f:
			    content = f.readlines()
			# you may also want to remove whitespace characters like `\n` at the end of each line
			content = [x.strip() for x in content] 

			for line in content:
				arr = line.split()
				try:
					sentiment = arr[1]
					sentence = ' '.join(arr[2:])
				except:
					exit('FUCKING ERROR')
				sentence = sentence.replace('"', "''")
				data = {'sentiment': sentiment, 'sentence': sentence}
				json.dump(data, outfile)
				outfile.write("\n")


def filter(token):
	if token[0] == '@':
		return '<at_@>'
	if token[:4] == 'http':
		return '<http>'
	return token.lower()


def filter_sentence(sent):
	return sent.replace('\\u2019', "'").replace('\\u002c', ",").replace('\"', '').replace('"', "'")


def func2():
	# read merged line by line, and tokenize
	merged_file = 'dataset_sentiment_not_target_semeval/dataset_sentiment_not_target/merged_dataset.json'
	fname_out = 'dataset_sentiment_not_target_semeval/dataset_sentiment_not_target/merged_dataset_processed.json'


	with open(merged_file, encoding="utf-8") as f:
	    content = f.readlines()
	# you may also want to remove whitespace characters like `\n` at the end of each line
	content = [x.strip() for x in content] 

	counter_dict = {'positive': 0, 'neutral': 0, 'negative': 0, 'ERROR_NO_LABEL': 0}

	with open(fname_out, 'w') as outfile:  
		for i, line in enumerate(content):

			obj = json.loads(line) 
			filtered_sentence = filter_sentence(obj['sentence'])
			tok_sent = [filter(tok.text) for tok in nlp.tokenizer(filtered_sentence)]

			data = {'post': tok_sent, 'label': obj['sentiment']}
			json.dump(data, outfile)
			outfile.write("\n")
			
			counter_dict[obj['sentiment']] += 1

	print(counter_dict)


def func1and2():
	files = [
		'twitter-2013dev-A.txt',
		'twitter-2013test-A.txt',
		'twitter-2013train-A.txt',
		'twitter-2014sarcasm-A.txt',
		'twitter-2014test-A.txt',
		'twitter-2015test-A.txt',
		'twitter-2015train-A.txt',
		'twitter-2016dev-A.txt',
		'twitter-2016devtest-A.txt',
		'twitter-2016test-A.txt',
		'twitter-2016train-A.txt',
	]

	prefix = 'dataset_sentiment_not_target_semeval/dataset_sentiment_not_target/'

	fname_out = prefix + 'merged_dataset.json'

	with open(fname_out, 'w') as outfile:  
		for filename in files:
			filepath = prefix + filename

			with open(filepath) as f:
			    content = f.readlines()
			# you may also want to remove whitespace characters like `\n` at the end of each line
			content = [x.strip() for x in content] 

			for line in content:
				arr = line.split()
				try:
					sentiment = arr[1]
					sentence = ' '.join(arr[2:])
				except:
					exit('FUCKING ERROR')

				filtered_sentence = filter_sentence(sentence)
				tok_sent = [filter(tok.text) for tok in nlp.tokenizer(filtered_sentence)]

				data = {'sentiment': sentiment, 'post': tok_sent}
				json.dump(data, outfile)
				outfile.write("\n")


# func1()
# func2()
# func1and2()