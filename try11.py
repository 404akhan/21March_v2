# import json


# fname = 'my_data_v4/combined_train_set.json'
# fname = 'my_data_v4/non_target_test.json'
# fname = 'my_data_v4/target_test.json'


# with open(fname) as f:
#     content = f.readlines()
# # you may also want to remove whitespace characters like `\n` at the end of each line
# content = [x.strip() for x in content] 

# print(content[0])


# json1_data = json.loads(content[0])

# print(json1_data['sentiment'])


counter = {}

def add(counter, name):
	if name in counter:
		counter[name] += 1
	else:
		counter[name] = 1

# for item in content:
# 	json1_data = json.loads(item)

# 	name = json1_data['sentiment']

# 	add(counter, name)

# print(counter)


fname = 'data-ANDREW-classified-pretty-v3.2.json'

import json

with open(fname) as f:
    data = json.load(f)

print(data[0])

for item in data:
	add(counter, item['NEW_predicted_sentiment'])

print(counter)