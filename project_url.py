#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 17:38:19 2022

@author: liuyilouise.xu
"""

import pandas as pd
import numpy as np
import ast
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize, wordpunct_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.cluster import KMeans

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

# remove retweets
def remove_retweet(df):
    no_retweets = df.loc[df['text'].str.find('RT @')==-1]
    print("no retweets: {}\n".format(np.shape(no_retweets)))
    return no_retweets

# get only tweets that contain url
# or return the binary feature vector `contain_url`
def keep_url(df, return_bool = False):
    # look for urls: return True if entities contains urls, else False. return False if entities = Nan
    url_bool = df['entities'].apply(lambda x:'urls' in ast.literal_eval(x).keys() if pd.notnull(x) else False)
    urls = df[url_bool]
    print("only contain urls: {}\n".format(np.shape(urls))) # tweets with url: 49523
    if return_bool:
        return url_bool
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

twitter = pd.read_csv("twitter_wildfire_data_oct22.csv", header=0)

#geo = twitter['geo']
#np.where(~pd.isnull(geo)) # only 830 entries with {} in geo col

# drop columns with no info
twitter = twitter.drop(labels='geo', axis=1)
twitter = twitter.drop(labels='media', axis=1)
twitter = twitter.drop(labels='author', axis=1)

# apply filters
twitter = remove_en(twitter)
twitter = remove_duplicate(twitter)
unrelated_words = ["CSGO", "covid"]
twitter = remove_keyword(twitter, unrelated_words)
twitter = remove_retweet(twitter)

# url analysis to further clean data
twitter_url = keep_url(twitter)
twitter_only_url = only_url(twitter_url)
url_fd_clean = get_url_fd(twitter_only_url)

# look at top 50 words
url_list = list(url_fd_clean.items())
url_list[0:50]

# url_fd_clean['org'] = 1769
# select url key words based on freq table
url_key_words= ["news", "national", "wildfire", "fire", "smoke"]
url_neglect_words= ["Miss_Wildfire", "clots", "cancers", "ryan", "cole"]

# remove tweets where url contains keyword
# or return binary feature vector of whether tweet url contains keyword
def remove_url_keyword(df, keywords, return_bool = False):
    
    def has_keyword(x, key_words):
        # false if tween doesn't have entity info
        if pd.isna(x):
            return False
        
        # false if tweet doesn't have url
        if 'urls' not in ast.literal_eval(x):
            return False
        
        # true if tweet url has key words
        url_list = ast.literal_eval(x)['urls']
        key_list = []
        for url in url_list:
            link = url.pop("unwound_url", None)
            if link == None:
                boolean = any(keyword in url['expanded_url'] for keyword in keywords)
            else:
                boolean = any(keyword in link for keyword in keywords)
            key_list.append(boolean)
        return any(key_list)
    
    # boolean for whether each tweet contains any of the url key words
    url_key_bool = df['entities'].apply(lambda x: has_keyword(x, keywords))
    if return_bool:
        return url_key_bool
    
    removed_url = df[~url_key_bool]
    print("remove key words in urls: {}\n".format(np.shape(removed_url)))
    return removed_url

def count_url(df):
    def count_url(x):
        # false if tween doesn't have entity info
        if pd.isna(x):
            return 0
        
        # false if tweet doesn't have url
        if 'urls' not in ast.literal_eval(x):
            return 0
        
        # true if tweet url has key words
        url_list = ast.literal_eval(x)['urls']
        return len(url_list)
    
    return df['entities'].apply(count_url)

# extract feature of whether text contains all caps words
def extract_caps(df):
    def has_caps(tweet):
        return len(re.findall(r'\b[A-Z]+\b', tweet)) != 0
    return df['text'].apply(has_caps)

def count_caps(df):
    def has_caps(tweet):
        return len(re.findall(r'\b[A-Z]+\b', tweet))
    return df['text'].apply(has_caps)

def extract_exclaim(df):
    def has_exclaim(tweet):
        return len(re.findall(r'\!', tweet)) != 0
    return df['text'].apply(has_exclaim)

def count_exclaim(df):
    def has_exclaim(tweet):
        return len(re.findall(r'\!', tweet))
    return df['text'].apply(has_exclaim)

def extract_hashtag(df):
    def has_hashtag(x):
        if pd.isna(x):
            return False
        if 'hashtags' in ast.literal_eval(x):
            return True
        return False
    return df['entities'].apply(has_hashtag)

def count_hashtag(df):
    def has_hashtag(x):
        if pd.isna(x):
            return 0
        if 'hashtags' in ast.literal_eval(x):
            return len(ast.literal_eval(x)['hashtags'])
        return 0
    return df['entities'].apply(has_hashtag)

# removed tweets with urls that contain neglect words
twitter = remove_url_keyword(twitter, url_neglect_words)

# binary feature of whether tweet contains url
contain_url = keep_url(twitter, True)
contain_url.name = 'url'

count_url = count_url(twitter)
count_url.name = 'count_url'


# all caps feature
contain_caps = extract_caps(twitter)
contain_caps.name = 'caps'

count_caps = count_caps(twitter)
count_caps.name = 'count_caps'

# exclamation mark feature
contain_exclaim = extract_exclaim(twitter)
contain_exclaim.name = 'exclaim'

count_exclaim = count_exclaim(twitter)
count_exclaim.name = 'count_exclaim'

# hash tag feature
contain_hashtag = extract_hashtag(twitter)
contain_hashtag.name = 'hashtag'

count_hashtag = count_hashtag(twitter)
count_hashtag.name = 'count_hashtag'

# url contain certain keywords
url_contain_keyword = remove_url_keyword(twitter, url_key_words, return_bool=True)
url_contain_keyword.name = 'url_keyword'

feature_df = pd.concat([count_url, 
                        url_contain_keyword,
                        count_caps,
                        count_exclaim,
                        count_hashtag], axis=1)



kmeans = KMeans(n_clusters=3).fit(feature_df)
tweet_labels = kmeans.labels_

'''
# analyze text to extract key word features
tokenized_text = []
for text in twitter['text']:
    tokenized_text += word_tokenize(text)

text_fd = nltk.FreqDist(tokenized_text)
text_fd = dict(text_fd)

# search for special characters
pattern = '[^\w\s]'
for key in list(text_fd.keys()):
    if re.search(pattern, key):
        text_fd.pop(key)
for key in list(text_fd.keys()):
    if key in stopwords.words("english"):
        text_fd.pop(key)

# list most common words in tweets
text_fd = dict(sorted(text_fd.items(), key=lambda item: item[1], reverse=True))
list(text_fd.items())[0:20]


from sklearn.feature_extraction.text import CountVectorizer
vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(1, 2))
X2 = vectorizer2.fit_transform(twitter['text'])
print(X2.toarray())
'''
