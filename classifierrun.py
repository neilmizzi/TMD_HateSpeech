from __future__ import print_function
import pandas as pd
from predicting.modifiedcode.onehotencoding import *
from keras.preprocessing import sequence
from keras.models import load_model

# load model
def lstm_predictions():
    DATA = pd.read_csv('predicting/modifiedcode/labeled_data (1).txt')
    LABELS = DATA['class'].values
    UNIQUE_LABELS = set(LABELS)

    DATA = pd.read_csv('tweets.csv')
    TWEETS = (DATA.loc[:, "tweet"]).tolist()

    maxlen = 280
    integer_mapped_tweets = [map_to_ints(t) for t in TWEETS]
    preprocessedtweets = sequence.pad_sequences(integer_mapped_tweets, maxlen=maxlen)

    model = load_model('predicting/modifiedcode/model.h5')

    predictions = model.predict_classes(preprocessedtweets)

    return predictions