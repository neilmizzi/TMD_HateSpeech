from numpy import array
import numpy as np
from keras.utils import to_categorical
import string

MAX_LENGTH = 280 #max lenght of a tweet
CHARS = list(string.printable) #100


def deal_with_unknown_characters(tweet):

    for char in tweet[:]:

        if char not in list(string.printable):


            tweet = tweet.replace(char,"")


    return tweet

def map_to_ints(a_string):

    integer_mapping = {x: i for i, x in enumerate(CHARS)}

    try:

        return [integer_mapping[char] for char in a_string]

    except:


        cleaned_up = deal_with_unknown_characters(a_string)

        return [integer_mapping[char] for char in cleaned_up]



#currently not used
# def get_one_hot(a_string):
#     integer_mapping = {x: i for i, x in enumerate(CHARS)}
#     string_vec = [integer_mapping[char] for char in a_string]
#     return to_categorical(string_vec, num_classes=len(CHARS))
#
# def pad_tweet(one_hot_encoded_tweet):
#     if len(one_hot_encoded_tweet) < MAX_LENGTH:
#         difference = MAX_LENGTH - len(one_hot_encoded_tweet)
#         zeros = np.zeros(shape=(difference, len(CHARS)))
#         return np.concatenate((one_hot_encoded_tweet,zeros))

