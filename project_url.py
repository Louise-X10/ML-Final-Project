#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 17:38:19 2022

@author: liuyilouise.xu
"""

import pandas as pd
import numpy as np
import ast
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize, wordpunct_tokenize, RegexpTokenizer
from nltk.corpus import stopwords

twitter = pd.read_csv("twitter_wildfire_data_oct22.csv", header=0)

#geo = twitter['geo']
#np.where(~pd.isnull(geo)) # only 830 entries with {} in geo col

# drop columns with no info
twitter = twitter.drop(labels='geo', axis=1)
twitter = twitter.drop(labels='media', axis=1)
twitter = twitter.drop(labels='author', axis=1)

# filter EN tweets only: 167494
twitter = twitter[(twitter['lang'] == "en")]

# filter unrelated words: 164388
unrelated_words = ["CSGO", "covid"]
twitter = twitter[~twitter['text'].str.contains('|'.join(unrelated_words))]

# look for urls: return True if entities contains urls, else False. return False if entities = Nan
url_bool = twitter['entities'].apply(
    lambda x:'urls' in ast.literal_eval(x).keys() if pd.notnull(x) else False)

# tweets with url: 49523
twitter_url = twitter[url_bool]

def return_expanded_url(x):
    url_list = ast.literal_eval(x)['urls']
    return_list = []
    for url in url_list:
        return_list.append(url['expanded_url'])
    return return_list

def return_unwound_url(x):
    url_list = ast.literal_eval(x)['urls']
    return_list = []
    for url in url_list:
        link = url.pop("unwound_url", None)
        if link == None:
            link = url['expanded_url']
        return_list.append(link)
    return return_list

# list of urls for each tweet
twitter_url_list = twitter_url["entities"].apply(return_unwound_url)

'''
temp_entity = twitter_url["entities"].iloc[0]
temp_url = ast.literal_eval(temp_entity)['urls'][0]
temp_url.keys()
temp_url['unwound_url']
'''

url_tokenizer = RegexpTokenizer(r'\w+')

url_words = []
for url_list in twitter_url_list:
    for url in url_list:
        url_words += url_tokenizer.tokenize(url)

# frequency table for words in url
url_fd = nltk.FreqDist(url_words)
del url_fd['https']
del url_fd['http']

# ignore words with less than 5 frequency
url_fd_clean = dict()
for key, value in dict(url_fd).items():
    if value > 100:
        url_fd_clean[key] = value

# sort by most frequent words
url_fd_clean = dict(sorted(url_fd_clean.items(), key=lambda item: item[1], reverse=True))

# look at top 30 words
sorted(url_fd_clean.items(), key=lambda item: item[1], reverse=True)[0:50]
# url_fd_clean['org'] = 1769

# select url key words based on freq table
url_key_words= ["news", "national", "wildfire", "fire", "smoke"]

url_neglect_words= ["Miss_Wildfire", "clots", "cancers", "ryan", "cole"]

def func(x, key_words):
    url_list = ast.literal_eval(x)['urls']
    key_list = []
    for url in url_list:
        boolean = any(key_word in url['expanded_url'] for key_word in key_words)
        key_list.append(boolean)
    return key_list

url_key_bool = twitter_url['entities'].apply(
    lambda x: func(x, url_key_words))

# boolean for whether each tweet contains url key words
url_key_bool_condense = url_key_bool.apply(any)

'''
d = ast.literal_eval(twitter_url['entities'].iloc[0])
len(d['urls']) # list of urls
for url in d['urls']:
    print(url.keys())
    url['expanded_url']

text = twitter.iloc[0,:]["text"]
words = [w for w in word_tokenize(text) if w.isalpha()]
fd = nltk.FreqDist(words)
'''

