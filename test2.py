# data preparation
import spacy
import json

nlp = spacy.load('en')


def get_data(fname):

	with open(fname) as f:
	    content = f.readlines()
	# you may also want to remove whitespace characters like `\n` at the end of each line
	content = [x.strip() for x in content] 

	text_arr = []
	label_arr = []
	for i in range(0, len(content), 3):
		text = content[i]
		target = content[i+1]
		label = int(content[i+2]) + 1

		text = text.replace("$T$", target)

		text_arr.append(text)
		label_arr.append(text)

	return text_arr, label_arr


# train_text_arr, train_label_arr = get_data('my_data/train.raw')
# test_text_arr, test_label_arr = get_data('my_data/test.raw')

# print(len(train_text_arr))

def get_label(c):
	if c == '-1':
		return 'negative'
	elif c == '1':
		return 'positive'
	elif c == '0':
		return 'neutral'
	else:
		return 'ERROR_SHOULDNT_REACH'


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

			post = post.replace("$T$", target)
			post = post.lower()
			tokenized = [tok.text for tok in nlp.tokenizer(post)]

			data = {'post': tokenized, 'label': label}
			json.dump(data, outfile)
			outfile.write("\n")


load_and_save('my_data/train.raw', 'my_data/train_prepared.txt')
load_and_save('my_data/test.raw', 'my_data/test_prepared.txt')
