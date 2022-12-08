import pandas as pd
import numpy as np
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

def is_positive(tweet):
    return 1 if sia.polarity_scores(tweet)['compound'] > 0 else 0

tweets = pd.read_csv('twitter_wildfire_data_oct22.csv')
tweets = tweets.loc[tweets['lang'] == 'en']
tweets = tweets.drop_duplicates(['text'])
tweet_text = [t.replace('://','//') for t in tweets['text']]

print('going through tweet polarity')
polarity = list(map(is_positive, tweet_text))
print('done going through tweet polarity')

tweets.insert(0, 'polarity', polarity)
tweets.to_csv('unique_tweet_sentiment.csv', mode='w', index=False)
