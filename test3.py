# do fastext on my data. Prepared data -> scapy -> fasttext -> predictions/accuracy

from torchtext import data
from torchtext import datasets

POST = data.Field()
LABEL = data.LabelField()


fields = {'post': ('p', POST), 'label': ('l', LABEL)}


train_data, test_data = data.TabularDataset.splits(
                            path = 'my_data',
                            train = 'train_prepared.txt',
                            test = 'test_prepared.txt',
                            format = 'json',
                            fields = fields
)

print(vars(train_data[0]))


import torch

SEED = 1234

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True




import random

train_data, valid_data = train_data.split(random_state=random.seed(SEED))


print(len(train_data))
print(len(valid_data))
print(len(test_data))


POST.build_vocab(train_data, max_size=25000)
LABEL.build_vocab(train_data)


print("Unique tokens in TEXT vocabulary: %d" % len(POST.vocab))
print("Unique tokens in LABEL vocabulary: %d" % len(LABEL.vocab))

print(POST.vocab.freqs.most_common(20))

print(POST.vocab.itos[:10])

print(LABEL.vocab.stoi)


BATCH_SIZE = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data), 
    batch_size=BATCH_SIZE,
    device=device)

