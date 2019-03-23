from torchtext import data
from torchtext import datasets
import torch

NAME = data.Field()
SAYING = data.Field()
PLACE = data.Field()

fields = {'name': ('n', NAME), 'location': ('p', PLACE), 'quote': ('s', SAYING)}

train_data, test_data = data.TabularDataset.splits(
                            path = 'sample_data',
                            train = 'sample_data_train.json',
                            test = 'sample_data_test.json',
                            format = 'json',
                            fields = fields
)

BATCH_SIZE = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


NAME.build_vocab(train_data)
SAYING.build_vocab(train_data)
PLACE.build_vocab(train_data)

train_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, test_data), sort_key=lambda x: len(x.n),
    batch_size=BATCH_SIZE,
    device=device)


for batch in train_iterator:
	print(batch)
for batch in test_iterator:
	print(batch)