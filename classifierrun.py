from __future__ import print_function
import pandas as pd
import numpy as np
from predicting.modifiedcode.onehotencoding import deal_with_unknown_characters, map_to_ints
from keras.preprocessing import sequence
from keras.models import load_model

# Please do not import tensorflow unnecessarily. Has severe impact on performance.

# load model
def lstm_predictions():
    DATA = pd.read_csv('datafiles/labeled_data.csv')
    LABELS = DATA['class'].values
    _ = set(LABELS)     # Why does this line exist if we are not using the variable?

    DATA = pd.read_csv('tweets.csv')
    TWEETS = (DATA.loc[:, "tweet"]).tolist()

    maxlen = 280
    integer_mapped_tweets = [map_to_ints(t) for t in TWEETS]
    preprocessedtweets = sequence.pad_sequences(integer_mapped_tweets, maxlen=maxlen)

    model = load_model('datafiles/model.h5', compile=False)

    predictions = model.predict_classes(preprocessedtweets)

    return predictions


def restructure_results(array):
    predictions = array.tolist()
    returnlist = []
    for prediction in predictions:
        if prediction == 0:
            returnlist.append('Hateful')
        if prediction == 1:
            returnlist.append('Offensive')
        if prediction == 2:
            returnlist.append('None')
    returnarray = np.asarray(returnlist)
    return returnarray
