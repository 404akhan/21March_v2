import spacy
from nltk import tokenize as nltk_tokenize_sentence
import pickle
import torch
import torch.nn as nn


def filter(token):
    if token[0] == '@':
        return 'atakhansspecialtoken'
    if token[:4] == 'http':
        return 'httpakhansspecialtoken'
    return token.lower()


nlp = spacy.load('en')
sentence = 'In terms of nominal GDP , Russia is only the twelfth biggest economy in the world httpakhansspecialtoken wiki List_of_countries_by_GDP_ nominal .'

tokenized = [filter(tok.text) for tok in nlp.tokenizer(sentence)]
tokenized = ' '.join(tokenized)
tokenized = [filter(tok.text) for tok in nlp.tokenizer(sentence)]
tokenized = ' '.join(tokenized)
tokenized = [filter(tok.text) for tok in nlp.tokenizer(sentence)]

print(tokenized)
print(' '.join(tokenized))