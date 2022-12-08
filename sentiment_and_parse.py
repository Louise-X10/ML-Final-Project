import pandas as pd
import numpy as np
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import sent_tokenize, word_tokenize

# author: adrian muth (using francisco's sentiment analyzer)
# vers: 12/8/22
# outputs a tweet's polarity and the frequency of each kind of token

polarity = pd.read_csv('unique_tweet_sentiment.csv', usecols = ['polarity'])

text = pd.read_csv('unique_tweet_sentiment.csv', usecols = ['text'])

print(np.shape(text))
print(np.shape(polarity))

# to be reviewed
grammar = """
NP: {<DT>?<JJ>*<NN>} #To extract Noun Phrases
P: {<IN>}            #To extract Prepositions
V: {<V.*>}           #To extract Verbs
PP: {<p> <NP>}       #To extract Prepositional Phrases
VP: {<V> <NP|PP>*}   #To extract Verb Phrases
"""
                    
chunk_parser = nltk.RegexpParser(grammar)

count = 0
for tweet in text.values:
    print(tweet)
    tokens = word_tokenize(np.array2string(tweet))
    pos_tags = nltk.pos_tag(tokens)
    tag_dict = {}
    for tag in pos_tags:
        if tag[1] not in tag_dict:
            tag_dict[tag[1]] = 1;
        else:
            tag_dict[tag[1]] += 1;
    print(tag_dict)
    print ("polarity: " + str(polarity.values[count]))
    count += 1
