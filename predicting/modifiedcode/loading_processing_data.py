import pandas as pd
import string
import itertools

CHARS = list(string.printable) # CONSTANT FOR POSSIBLE CHARACTERS

def deal_with_unknown_characters(tweet):
    for char in tweet[:]:
        if char not in CHARS:
            tweet = tweet.replace(char,"")
    return tweet

def map_to_ints(a_string):
    integer_mapping = {x: i for i, x in enumerate(CHARS)}
    try:
        return [integer_mapping[char] for char in a_string]
    except:
        cleaned_up = deal_with_unknown_characters(a_string)
        return [integer_mapping[char] for char in cleaned_up]

#LOADING

# training and evaluation data
TRAIN_DATA = pd.read_csv('datafiles/Data-Set 20_20_60.csv',sep = '\t') #change this file to try different splits
TRAIN_TWEETS = TRAIN_DATA['0'].astype(str).values # training data
TRAIN_LABELS = TRAIN_DATA['1'].values
TRAIN_UNIQUE_LABELS = set(TRAIN_LABELS)

#predict data - replace first none with pd.read_csv(filepath), second none with column name
PREDICT_DATA, TWEET_COLUMN = None, None
#PREDICT_TWEETS = PREDICT_DATA[TWEET_COLUMN].astype(str).values #comment out when filled in prediction data

#TRANSFORMING

# integer mapping for the train data
integer_mapped_tweets = [map_to_ints(t) for t in TRAIN_TWEETS]
unique_characters = merged = list(itertools.chain.from_iterable(integer_mapped_tweets)) #flattening nested list: making one list out of all the lists so we can use set()

# check how many letters their are in the tweets
print("Unique characters:",len(set(unique_characters)))

