import pandas as pd
import numpy as np
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import sent_tokenize, word_tokenize

# author: adrian muth (using francisco's sentiment analyzer)
# vers: 12/8/22
# outputs a tweet's polarity and the frequency of each kind of token

# grab data
polarity = pd.read_csv('unique_tweet_sentiment.csv', usecols = ['polarity'])
text = pd.read_csv('unique_tweet_sentiment.csv', usecols = ['text'])

print(np.shape(text))
print(np.shape(polarity))

# louise's grammar for parsing tweets
grammar = """
NP: {<DT>?<JJ>*<NN>} #To extract Noun Phrases
P: {<IN>}            #To extract Prepositions
V: {<V.*>}           #To extract Verbs
PP: {<p> <NP>}       #To extract Prepositional Phrases
VP: {<V> <NP|PP>*}   #To extract Verb Phrases
"""
                    
chunk_parser = nltk.RegexpParser(grammar)

# find word tag frequency and polarity for each tweet
# output results
polar_one_count = 0 # amount of positive tweets
count = 0
zero_dict = {}
one_dict = {}
total_dict = {}
for tweet in text.values:
    print(tweet)
    tokens = word_tokenize(np.array2string(tweet))
    pos_tags = nltk.pos_tag(tokens)
    tweet_pol = polarity.values[count]
    tag_dict = {}
    # modify coordinating dictionary and return modifications back
    if tweet_pol == 1:
        polar_one_count += 1
        tag_dict = one_dict
    else:
        tag_dict = zero_dict
    for tag in pos_tags:
        if tag[1] not in tag_dict:
            tag_dict[tag[1]] = 1;
        else:
            tag_dict[tag[1]] += 1;
    print(tag_dict)
    print("polarity: " + str(tweet_pol))
    count += 1
    if tweet_pol == 1:
        one_dict = tag_dict
    else:
        zero_dict = tag_dict

# output
print(polar_one_count)
print(zero_dict)
print(one_dict)
