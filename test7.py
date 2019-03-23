# create datasets

import csv

with open('my_data_v2/training.1600000.processed.noemoticon.csv') as csv_file:
	csv_read = csv.reader(csv_file, delimiter=',')


print(csv_read[0])