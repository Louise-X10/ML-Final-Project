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

#remove entries about CSGO
def remove_CSGO(df):
    no_csgo = df.loc[~df['text'].str.contains('CSGO', regex=False)]
    print("without csgo tweets: {}\n".format(np.shape(no_csgo)))
    return no_csgo

def remove_keyword(df, keywords):
# filter unrelated words: 164388
    no_keywords = df[~df['text'].str.contains('|'.join(keywords), regex=False)]
    return no_keywords

#remove entries not in english
def remove_en(df):
    only_en = df.loc[df['lang'] == 'en']
    print("tweets with 'en' as language: {}\n".format(np.shape(only_en)))
    return only_en

#get only unique tweet text
def remove_duplicate(df):
    unique = df.drop_duplicates(['text'])
    print("unique entries in text column: {}\n".format(np.shape(unique)))
    return unique

#get only retweets
def keep_retweet(df):
    retweets = df.loc[df['text'].str.find('RT @')==0]
    print("only retweets: {}\n".format(np.shape(retweets)))
    return retweets

# get only tweets that contain url
def keep_url(df):
    # look for urls: return True if entities contains urls, else False. return False if entities = Nan
    url_bool = df['entities'].apply(lambda x:'urls' in ast.literal_eval(x).keys() if pd.notnull(x) else False)
    urls = df[url_bool]
    print("only contain urls: {}\n".format(np.shape(urls))) # tweets with url: 49523
    return urls

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

# get column of urls
def only_url(df):
    only_url = df["entities"].apply(return_unwound_url)
    return only_url

# get frequency table for words in url
# option to ignore words with less than cutoff frequency
def get_url_fd (df, clean = True, cutoff = 100):
    url_tokenizer = RegexpTokenizer(r'\w+')
    
    url_words = []
    for url_list in df:
        for url in url_list:
            url_words += url_tokenizer.tokenize(url)
    
    url_fd = nltk.FreqDist(url_words)
    del url_fd['https']
    del url_fd['http']
    
    if clean:
        url_fd_clean = dict()
        for key, value in dict(url_fd).items():
            if value > cutoff:
                url_fd_clean[key] = value
        # sort by highest freq
        url_fd_clean = dict(sorted(url_fd_clean.items(), key=lambda item: item[1], reverse=True))
        return url_fd_clean
    
    return url_fd
    
# apply filters
twitter = remove_en(twitter)
twitter = remove_duplicate(twitter)
unrelated_words = ["CSGO", "covid"]
twitter = remove_keyword(twitter, unrelated_words)

# url analysis to further clean data
twitter_url = keep_url(twitter)
twitter_only_url = only_url(twitter_url)
url_fd_clean = get_url_fd(twitter_only_url)

# look at top 50 words
url_list = list(url_fd_clean.items())
url_list[0:50]

# url_fd_clean['org'] = 1769
# select url key words based on freq table
# url_key_words= ["news", "national", "wildfire", "fire", "smoke"]
url_neglect_words= ["Miss_Wildfire", "clots", "cancers", "ryan", "cole"]

def remove_url_keyword(df, keywords):
    
    def func(x, key_words):
        url_list = ast.literal_eval(x)['urls']
        key_list = []
        for url in url_list:
            link = url.pop("unwound_url", None)
            if link == None:
                boolean = any(key_word in url['expanded_url'] for key_word in key_words)
            else:
                boolean = any(key_word in link for key_word in key_words)
            key_list.append(boolean)
        return key_list

    url_key_bool = df['entities'].apply(lambda x: func(x, keywords))

    # boolean for whether each tweet contains any of the url key words
    url_key_bool_condense = url_key_bool.apply(any)
    return df[~url_key_bool_condense]

twitter_url = remove_url_keyword(twitter_url, url_neglect_words)
