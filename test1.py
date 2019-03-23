import preprocessor as p
import re
import spacy
nlp = spacy.load('en')

sentence = "haven't can't would've could've don't hello my Lord! how r u doin,Bro:)" #' https://www.facebook.com/'

tokenized = [tok.text for tok in nlp.tokenizer(sentence)]

print(tokenized)


tok = nlp.tokenizer(sentence)[0]

print(tok)
print(tok.text)


import json

with open('data.json') as f:
    data = json.load(f)


for i, item in enumerate(data):
    print('\n\n')
    tokenized = [tok.text for tok in nlp.tokenizer(item['body'])]
    print(tokenized)

    if i >= 10:
        break
