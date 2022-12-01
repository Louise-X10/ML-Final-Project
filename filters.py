#author: Francisco Nguyen
#some examples of filtering the tweets

import pandas as pd
import numpy as np

df = pd.read_csv('twitter_wildfire_data_oct22.csv')
print(df.columns)
print("\noriginal shape of data: {}\n".format(np.shape(df)))

#remove entries about CSGO
no_csgo = df.loc[~df['text'].str.contains('CSGO', regex=False)]
print("without csgo tweets: {}\n".format(np.shape(no_csgo)))

#remove entries not in english
only_en = df.loc[df['lang'] == 'en']
print("tweets with 'en' as language: {}\n".format(np.shape(only_en)))

#get only unique tweet text
unique = df.drop_duplicates(['text'])
print("unique entries in text column: {}\n".format(np.shape(unique)))

#get only retweets
retweets = df.loc[df['text'].str.find('RT @')==0]
print("only retweets: {}\n".format(np.shape(retweets)))

"""
data = np.array(df.values)
print(data.shape)
data = data[np.where('CSGO Skin Giveaway' in data[:,1])]
print(data.shape)
"""
