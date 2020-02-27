import twint
import csv
import pandas as pd
import copy
import string
import nest_asyncio
from datetime import datetime
from preprocessing import PreProcessing



class TwintHandler:
    # Dataframe layout in https://github.com/twintproject/twint/wiki/Tweet-attributes#attributes

    c = twint.Config()                  # Config File
    tweets = pd.DataFrame()             # Unmodified Tweets (Full DataFrame)
    cleaned_tweets = pd.DataFrame()     # Cleaned Tweets (Full DataFrame)


    def search_user(self, user, tweet_limit = 1000, 
    date_lim_lower = '2007-01-01 00:00:00', 
    date_lim_upper = datetime.now().strftime("%Y-%m-%d %H:%M:%S")):            
        self.c.Limit = tweet_limit              # No. of Tweet Limit
        self.c.Hide_output = True               # Hide Output
        self.c.Username = user                  # Username to look up
        self.c.Pandas = True                    # Set config to return Pandas DataFrame
        self.c.Until = date_lim_upper           # Upper date limit bound
        self.c.Since = date_lim_lower           # Lower date limit bound
        twint.run.Search(self.c)                # Run Search

        # Return tweets
        self.tweets = twint.storage.panda.Tweets_df
        return self.tweets

    # Return Pre-Processed Tweets
    def get_preprocessed_tweets(self):
        prep = PreProcessing()
        self.cleaned_tweets = copy.deepcopy(self.tweets)
        for i in range(0, tweets.shape[0]):
            self.cleaned_tweets.at[i, 'tweet'] = prep.preprocess(self.cleaned_tweets.at[i, 'tweet'])
        return self.cleaned_tweets


# USAGE CODE (Example on how to extract tweets etc)
handler = TwintHandler()
tweets = handler.search_user('realDonaldTrump', tweet_limit=10)
cleaned_tweets = handler.get_preprocessed_tweets()

print(tweets.head())
print(cleaned_tweets['tweet'])