import twint
import csv
import pandas as pd
import copy
import string
import nest_asyncio
from datetime import datetime
from preprocessing.preprocessing import PreProcessing
import os



class TwintHandler:
    # Dataframe layout in https://github.com/twintproject/twint/wiki/Tweet-attributes#attributes
    tweets = pd.DataFrame()             # Unmodified Tweets (Full DataFrame)
    cleaned_tweets = pd.DataFrame()     # Cleaned Tweets (Full DataFrame)


    def search_user(self, user, tweet_limit = 1000, date_lim_lower = '2007-01-01 00:00:00', date_lim_upper = datetime.now().strftime("%Y-%m-%d %H:%M:%S")):
        # Set defaults in case of blank fields
        if os.path.exists('./tweets.csv'):
            os.remove('./tweets.csv')
        if user == '':
            raise Exception
        if tweet_limit == 0:
            tweet_limit = 100
        if date_lim_lower == '':
            date_lim_lower = '2007-01-01 00:00:00'
        if date_lim_upper == '':
            date_lim_upper = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c = twint.Config()                  # Config File
        c.Limit = int(tweet_limit)          # No. of Tweet Limit
        c.Hide_output = True                # Hide Output
        c.Username = user                   # Username to look up

        # c.Pandas = True                   # Set config to return Pandas DataFrame
        c.Store_csv = True
        c.Output = 'tweets.csv'

        c.Until = date_lim_upper            # Upper date limit bound
        c.Since = date_lim_lower            # Lower date limit bound
        twint.run.Search(c)                 # Run Search

        # Return tweets
        if not(os.path.exists('./tweets.csv')):
            raise Exception
        self.tweets = pd.read_csv('tweets.csv')
        return self.tweets

    # Return Pre-Processed Tweets
    def get_preprocessed_tweets(self):
        prep = PreProcessing()
        self.cleaned_tweets = copy.deepcopy(self.tweets)
        for i in range(0, self.tweets.shape[0]):
            self.cleaned_tweets.at[i, 'tweet'] = prep.preprocess(self.cleaned_tweets.at[i, 'tweet'])
        return self.cleaned_tweets


# USAGE CODE (Example on how to extract tweets etc)
"""
handler = TwintHandler()
tweets = handler.search_user('realDonaldTrump', tweet_limit=10)
cleaned_tweets = handler.get_preprocessed_tweets()

print(tweets.head())
print(cleaned_tweets['tweet'])
"""
