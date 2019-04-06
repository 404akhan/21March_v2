import pickle 
import json 

with open('data-ANDREW.json') as f:
	data = json.load(f)


for i, item in enumerate(data):
	text = item['body'].lower()
	target = ' '.join(item['name'])
	text = text.replace(target, 'my_target_wrapper ' + target + ' my_target_wrapper')

	print(text)
