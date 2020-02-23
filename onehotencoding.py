from numpy import array
import numpy as np
from keras.utils import to_categorical
import string

MAX_LENGTH = 280 #max lenght of a tweet
CHARS = list(string.printable)

def get_one_hot(a_string):
    integer_mapping = {x: i for i, x in enumerate(CHARS)}
    string_vec = [integer_mapping[char] for char in a_string]
    return to_categorical(string_vec, num_classes=len(CHARS))

def pad_tweet(one_hot_encoded_tweet):
    if len(one_hot_encoded_tweet) < MAX_LENGTH:
        difference = MAX_LENGTH - len(one_hot_encoded_tweet)
        zeros = np.zeros(shape=(difference, len(CHARS)))
        return np.concatenate((one_hot_encoded_tweet,zeros))


