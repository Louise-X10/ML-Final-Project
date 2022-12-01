import pandas as pd
import numpy as np
import nltk
from nltk import sent_tokenize, word_tokenize

# author: adrian muth
# vers: 12/1/22
# generates a structural parse tree for every unique, english tweet

data = pd.read_csv("twitter_wildfire_data_oct22.csv")

#remove entries not in english
only_en = data.loc[data['lang'] == 'en']
print("tweets with 'en' as language: {}\n".format(np.shape(only_en)))

#get only unique tweet text
unique = only_en.drop_duplicates(['text'])
print("unique entries in text column: {}\n".format(np.shape(unique)))

unique = unique.drop(labels='geo', axis=1)
unique = unique.drop(labels='media', axis=1)
unique = unique.drop(labels='author', axis=1)

text = unique['text']

# to be reviewed
grammar = "NP: {<DT>?<JJ>*<NN>}"
chunk_parser = nltk.RegexpParser(grammar)

for tweet in text:
    print(tweet)
    tokens = word_tokenize(tweet)
    print(tokens)
    pos_tags = nltk.pos_tag(tokens)
    print(pos_tags)
    tree = chunk_parser.parse(pos_tags)
    tree.draw()
