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
from random import shuffle

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
    no_keywords = df[~df['text'].str.contains('|'.join(keywords), regex=True)]
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
unrelated_words = ["CSGO", "covid"] # tweets with these words are considered unrelated even if they contain words like "wildfire"
twitter = remove_keyword(twitter, unrelated_words)

# url analysis to further trim dataset
url_fd = get_url_fd(twitter)
hashtag_fd = get_hashtag_fd(twitter)

# look at most frequent url words, hashtags
url_fd.most_common(20)
hashtag_fd.most_common(20)

# select url key words based on freq table
url_keywords= ["news", "national", "wildfire", "fire", "smoke"] 
url_neglect_words= ["Miss_Wildfire", "clots", "cancers", "ryan", "cole"] 


# remove tweets where url contains url_neglect_words
def remove_url_keyword(df, keywords):
    url_key_bool = extract_url_keyword(df, keywords)
    removed_url = df[~url_key_bool]
    print("remove key words in urls: {}\n".format(np.shape(removed_url)))
    return removed_url

# extract binary feature vector: whether tweet url contains any of the url_keywords
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

# extract binary feature vector: whether tweet contains url
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

# extract numeric feature: number of url a tweet contains
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

# extract binary feature: whether text contains all caps words 
def extract_caps(df):
    def has_caps(tweet):
        # capture words in all caps with at least length 2
        return len(re.findall(r'\b[A-Z]{2,}\b', tweet)) != 0
    return df['text'].apply(has_caps)

# extract numeric feature: number of all caps words in tweet
# for example, tweet 40387: "THIS BANGER TWEET SO TRU LOUDER FOR THE PEOPLE IN THE BACK SPREAD THIS LIKE WILDFIRE"
# has count = 16
# but also have to beware of propoer nouns like AR in tweet 3
def count_caps(df):
    def has_caps(tweet):
        return len(re.findall(r'\b[A-Z]+\b', tweet))
    return df['text'].apply(has_caps)

# extract binary feature: whether tweet contains exclamation mark
def extract_exclaim(df):
    def has_exclaim(tweet):
        return len(re.findall(r'\!', tweet)) != 0
    return df['text'].apply(has_exclaim)

# extract numeric feature: number of exclamation marks a tweet contains
def count_exclaim(df):
    def has_exclaim(tweet):
        return len(re.findall(r'\!', tweet))
    return df['text'].apply(has_exclaim)

# extract binary hashtag feature: whether tweet contains hashtags
def extract_hashtag(df):
    def has_hashtag(x):
        if pd.isna(x):
            return False
        if 'hashtags' in ast.literal_eval(x):
            return True
        return False
    return df['entities'].apply(has_hashtag)

# extract numeric hashtag feature: number of hashtags a tweet contains
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

# insert sentiment labels using pre-trained nltk model
sia = SentimentIntensityAnalyzer()

def is_positive(tweet):
    return 1 if sia.polarity_scores(tweet)['compound'] > 0 else 0

tweet_text = [t.replace('://','//') for t in twitter['text']]

print('going through tweet polarity')
polarity = list(map(is_positive, tweet_text))
print('done going through tweet polarity')

twitter.insert(0, 'polarity', polarity)

# compute url features
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

# whether url contain certain keywords
url_contain_keyword = extract_url_keyword(twitter, url_keywords)
url_contain_keyword.name = 'url_keyword'

# next hashtag analysis
# look at most common hashtags
hashtag_fd.most_common(40)

# extract 40 most common hashtags
hashtag_keywords = list(zip(*hashtag_fd.most_common(40)))[0]
hashtag_keywords = list(hashtag_keywords)

# extract numeric hashtag feature: the number of common hashtags that a tweet contains
def extract_common_hashtag_count(df, keywords):
    # get column of tag lists
    only_tags = df['entities'].apply(return_hashtags)
    
    # count number of common tags contained in each tweet
    common_tags = only_tags.apply(lambda tag_list: sum(1 for tag in tag_list if tag in keywords))
    
    return common_tags

# common hashtag feature
count_hashtag_common = extract_common_hashtag_count(twitter, hashtag_keywords)
count_hashtag_common.name = 'count_hashtag_common'

# extract word type count
def count_wordtype(df):
    text = df['text']
    polarity = df['polarity']
    polarity.reset_index(drop=True,inplace=True)
    count_df = pd.DataFrame(polarity)
    for i, tweet in enumerate(text):
        tokens = word_tokenize(tweet)
        pos_tags = nltk.pos_tag(tokens)
        for tag in pos_tags:
            if tag[1] not in count_df.columns:
                new_col = pd.DataFrame(np.zeros(len(polarity)), columns=[tag[1]])
                count_df= pd.concat([count_df, new_col], axis=1)
                count_df[tag[1]].iloc[i] = 1
            else:
                count_df[tag[1]].iloc[i] += 1
    count_df.drop(labels="polarity", axis=1)
    return count_df
    
    
wordtype_count = count_wordtype(twitter)


### MODEL 1 feature vectors
feature_df = pd.concat([count_url, 
                        url_contain_keyword,
                        count_caps,
                        count_exclaim,
                        count_hashtag,
                        count_hashtag_common], axis=1)

feature_df.reset_index(drop=True,inplace=True)
feature_df = pd.concat([feature_df, wordtype_count], axis=1)

# format features for NLTK Naive Bayes classifer
def extract_features(tweet_data):
    features = dict()
    for idx in tweet_data.index:
        features[idx] = tweet_data[idx]
    return features

features = [(extract_features(t[1]), p)
            for t, p in zip(feature_df.iterrows(), polarity)]

# train naive bayes
train_count = int(len(features) * 0.8)
shuffle(features)
classifier = nltk.NaiveBayesClassifier.train(features[:train_count])

# find most informative features
classifier.show_most_informative_features(10)
print('Validation accuracy: {}'.format(
    nltk.classify.accuracy(classifier, features[train_count:])))

'''
### MODEL 2: this model is not used becuase it's information can be consolidated into one feature vector `count_hashtag_common`

# returns a dataframe, each column is a common hashtag, encodes the frequency of that hashtag for each tweet
def extract_common_hashtag_features(df, keywords):
    # get column of tag lists
    only_tags = df['entities'].apply(return_hashtags)
    
    count_df = pd.DataFrame(np.zeros((only_tags.shape[0], len(keywords))), columns = keywords)
    
    for i, tag_list in enumerate(only_tags):
        for tag in tag_list:
            if tag in keywords:
                count_df[tag].iloc[i] += 1
                
    return count_df

# common hashtag frequency for each tweet
hashtag_df = extract_common_hashtag_features(twitter, hashtag_keywords)
'''
