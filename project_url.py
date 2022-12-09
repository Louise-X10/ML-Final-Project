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
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import PorterStemmer

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

# remove retweets
def remove_retweet(df):
    no_retweets = df.loc[df['text'].str.find('RT @')==-1]
    print("no retweets: {}\n".format(np.shape(no_retweets)))
    return no_retweets

#remove entries containing keywords
def remove_keyword(df, keywords):
    no_keywords = df[~df['text'].str.contains('|'.join(keywords), regex=False)]
    print("tweets with keywords removed: {}\n".format(np.shape(no_keywords)))
    return no_keywords

# returns list of expanded urls for each tweet
def return_expanded_url(x):
    if pd.isna(x):
        return []
    if 'urls' not in ast.literal_eval(x):
            return []
    url_list = ast.literal_eval(x)['urls']
    return_list = []
    for url in url_list:
        return_list.append(url['expanded_url'])
    return return_list

# returns list of unwound urls for each tweet
def return_unwound_url(x):
    if pd.isna(x):
        return []
    if 'urls' not in ast.literal_eval(x):
            return []
    url_list = ast.literal_eval(x)['urls']
    return_list = []
    for url in url_list:
        link = url.pop("unwound_url", None)
        if link == None:
            link = url['expanded_url']
        return_list.append(link)
    return return_list

# get frequency table for words in url
def get_url_fd (df):
    
    # get column of url lists
    only_url = df["entities"].apply(return_unwound_url)
    
    url_tokenizer = RegexpTokenizer(r'\w+')
    
    url_words = []
    for url_list in only_url:
        for url in url_list:
            url_words += url_tokenizer.tokenize(url)
    
    # keep words less than length 15, not stop words
    url_clean_words = []
    for word in url_words:
        word = word.lower()
        if not re.match('^\w{0,15}$', word):
            continue
        if word in stopwords.words("english"):
            continue
        url_clean_words.append(word)
    
    url_fd = nltk.FreqDist(url_clean_words)
    del url_fd['https']
    del url_fd['http']
    del url_fd['www']
    del url_fd['com']
    
    return url_fd

# returns list of hashtags for each tweet
def return_hashtags(x):
    if pd.isna(x):
        return []    
    if 'hashtags' not in ast.literal_eval(x):
        return []    
    hashtag_list = ast.literal_eval(x)['hashtags']
    key_list = []
    for hashtag in hashtag_list:
        key_list.append(hashtag['tag'].lower())
    return key_list
    
# returns frequency of hashtags
def get_hashtag_fd (df):

    only_tags = df['entities'].apply(return_hashtags)
    
    all_tags = []
    for tag in only_tags:
        all_tags += tag
    
    # keep words less than length 15, not stop words, stemmed
    hashtag_words = []
    for word in all_tags:
        if not re.match('^\w{0,15}$', word):
            continue
        if word in stopwords.words("english"):
            continue
        hashtag_words.append(word)
    
    hashtag_fd = nltk.FreqDist(hashtag_words)
    
    return hashtag_fd

twitter = pd.read_csv("twitter_wildfire_data_oct22.csv", header=0)

# drop columns with no info
twitter = twitter.drop(labels='geo', axis=1)
twitter = twitter.drop(labels='media', axis=1)
twitter = twitter.drop(labels='author', axis=1)

# apply filters
twitter = remove_en(twitter)
twitter = remove_duplicate(twitter)
twitter = remove_retweet(twitter)
unrelated_words = ["CSGO", "covid"]
twitter = remove_keyword(twitter, unrelated_words)

# url analysis 
url_fd = get_url_fd(twitter)
hashtag_fd = get_hashtag_fd(twitter)

# look at most frequent url words
url_fd.most_common(20)
hashtag_fd.most_common(20)

# url_fd_clean['org'] = 1769
# select url key words based on freq table
url_keywords= ["news", "national", "wildfire", "fire", "smoke"]
url_neglect_words= ["Miss_Wildfire", "clots", "cancers", "ryan", "cole"]

# remove tweets where url contains keyword
# or return binary feature vector of whether tweet url contains keyword

def extract_url_keyword(df, keywords):
    # get column of url lists
    only_url = df["entities"].apply(return_unwound_url)
    
    def has_keyword(url_list, keywords):
        key_list = []
        for url in url_list:
            boolean = any(keyword in url for keyword in keywords)                
            key_list.append(boolean)
        return any(key_list)
    
    # boolean for whether each tweet contains any of the url key words
    url_key_bool = only_url.apply(lambda x: has_keyword(x, keywords))
    return url_key_bool
    
def remove_url_keyword(df, keywords):
    url_key_bool = extract_url_keyword(df, keywords)
    removed_url = df[~url_key_bool]
    print("remove key words in urls: {}\n".format(np.shape(removed_url)))
    return removed_url

# extract url feature
def extract_url(df):
    def count_url(x):
        # false if tween doesn't have entity info
        if pd.isna(x):
            return False
        
        # false if tweet doesn't have url
        if 'urls' in ast.literal_eval(x):
            return True
        else:
            return False
    
    return df['entities'].apply(count_url)

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

# extract exclamation mark feature
def extract_exclaim(df):
    def has_exclaim(tweet):
        return len(re.findall(r'\!', tweet)) != 0
    return df['text'].apply(has_exclaim)

def count_exclaim(df):
    def has_exclaim(tweet):
        return len(re.findall(r'\!', tweet))
    return df['text'].apply(has_exclaim)

# extract hashtag feature
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
# twitter = remove_url_keyword(twitter, url_neglect_words)

# insert sentiment labels
sia = SentimentIntensityAnalyzer()

def is_positive(tweet):
    return 1 if sia.polarity_scores(tweet)['compound'] > 0 else 0

tweet_text = [t.replace('://','//') for t in twitter['text']]

print('going through tweet polarity')
polarity = list(map(is_positive, tweet_text))
print('done going through tweet polarity')

twitter.insert(0, 'polarity', polarity)

# binary feature of whether tweet contains url
contain_url = extract_url(twitter)
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
url_contain_keyword = extract_url_keyword(twitter, url_keywords)
url_contain_keyword.name = 'url_keyword'

# most common hashtags
hashtag_fd.most_common(40)
hashtag_keywords = list(zip(*hashtag_fd.most_common(40)))[0]
hashtag_keywords = list(hashtag_keywords)

def extract_common_hashtag_count(df, keywords):
    # get column of tag lists
    only_tags = df['entities'].apply(return_hashtags)
    
    # count number of common tags for each tweet
    common_tags = only_tags.apply(lambda tag_list: sum(1 for tag in tag_list if tag in keywords))
    
    return common_tags

# common hashtag feature
count_hashtag_common = extract_common_hashtag_count(twitter, hashtag_keywords)
count_hashtag_common.name = 'count_hashtag_common'

### MODEL 1
feature_df = pd.concat([count_url, 
                        url_contain_keyword,
                        count_caps,
                        count_exclaim,
                        count_hashtag,
                        count_hashtag_common], axis=1)

def extract_common_hashtag_features(df, keywords):
    # get column of tag lists
    only_tags = df['entities'].apply(return_hashtags)
    
    count_df = pd.DataFrame(np.zeros((only_tags.shape[0], len(keywords))), columns = keywords)
    
    for i, tag_list in enumerate(only_tags):
        for tag in tag_list:
            if tag in keywords:
                count_df[tag].iloc[i] += 1
                
    return count_df

### MODEL 2
# common hashtag frequency for each tweet
hashtag_df = extract_common_hashtag_features(twitter, hashtag_keywords)

#kmeans = KMeans(n_clusters=3).fit(feature_df)
#tweet_labels = kmeans.labels_

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
