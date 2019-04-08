import json

with open('data-ANDREW-classified-pretty-v3.json') as data_file:    
    data = json.load(data_file)

counter = {}
for item in data:
	combined = item['body'] + '###' + ' '.join(item['target_arr'])
	if combined in counter:
		counter[combined] += 1
	else:
		counter[combined] = 1

count_good = 0
count_bad = 0
for key, val in counter.items():
	if val == 1:
		count_good += 1
	if val != 1:
		count_bad += 1
	# print(key, val)
print(count_good, count_bad)
print(count_bad / (count_good + count_bad))










exit(0)
print(len(data))

dict_counter = {'neutral': 0, 'negative': 0, 'positive': 0}
dict_counter_old = {'neutral': 0, 'negative': 0, 'positive': 0}
count_disagree = 0
count_agree = 0

def convert(sent):
	if sent == -1:
		return 'negative'
	if sent == 0:
		return 'neutral'
	if sent == 1:
		return 'positive'

for item in data:
	# print(item)
	dict_counter[item['NEW_predicted_sentiment']] += 1
	dict_counter_old[convert(item['old_sentiment'])] += 1

	if 	item['NEW_predicted_sentiment'] == convert(item['old_sentiment']):
		count_agree += 1
	else:
		count_disagree += 1


print(dict_counter)
print(dict_counter_old)
print(count_disagree, count_agree)


