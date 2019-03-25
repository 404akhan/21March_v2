import random


def split_merged():
	fname1 = 'my_data_v4/merged_dataset.json'
	with open(fname1) as f:
		content1 = f.readlines()
	content1 = [x.strip() for x in content1] 

	print(content1[0:4])

	random.shuffle(content1)
	train_len = int(len(content1) * 0.7)

	train_data = content1[:train_len]
	test_data = content1[train_len:]


	with open('my_data_v4/merged_dataset_test.json', 'w') as outfile:  
		for data_item in test_data:

			outfile.write(data_item)
			outfile.write("\n")



	fname2 = 'my_data_v4/old_twitter_train.json'
	with open(fname2) as f:
	    content2 = f.readlines()
	content2 = [x.strip() for x in content2] 

	combined_content = train_data + content2
	random.shuffle(combined_content)

	with open('my_data_v4/combined_train_set.json', 'w') as outfile:  
		for data_item in combined_content:

			outfile.write(data_item)
			outfile.write("\n")



def fix_label_sentiment(fname):
	with open(fname) as f:
	    content = f.readlines()

	with open(fname[:-5] + 'v2.json', 'w') as outfile:  
		for data_item in content:

			item_wr = data_item
			item_wr = item_wr.replace('"label": "negative"', '"sentiment": "negative"')
			item_wr = item_wr.replace('"label": "neutral"', '"sentiment": "neutral"')
			item_wr = item_wr.replace('"label": "positive"', '"sentiment": "positive"')

			outfile.write(item_wr)

fix_label_sentiment('combined_train_set.json')
fix_label_sentiment('target_test.json')
