import spacy

nlp = spacy.load('en')

def filter(token):
    if token[0] == '@':
        return '<at_@>'
    if token[:4] == 'http':
        return '<http>'
    return token.lower()

sent = 'enter [hi](http://facebook.com) to see latest updates!'

tokenized = [filter(tok.text) for tok in nlp.tokenizer(sent)]

print(sent)
print(tokenized)


import nltk.data

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

print(tokenizer.tokenize(sent))
# print('\n-----\n'.join(tokenizer.tokenize(sent)))


from nltk import tokenize
print(tokenize.sent_tokenize(sent))