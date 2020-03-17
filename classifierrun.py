from __future__ import print_function
import pandas as pd
from predicting.modifiedcode.onehotencoding import *
from keras.preprocessing import sequence
from keras.models import load_model
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# load model
def lstm_predictions():
    DATA = pd.read_csv('data sets/labeled_data.csv')
    LABELS = DATA['class'].values
    UNIQUE_LABELS = set(LABELS)

    DATA = pd.read_csv('tweets.csv')
    TWEETS = (DATA.loc[:, "tweet"]).tolist()

    maxlen = 280
    integer_mapped_tweets = [map_to_ints(t) for t in TWEETS]
    preprocessedtweets = sequence.pad_sequences(integer_mapped_tweets, maxlen=maxlen)

    model = load_model('data sets/model.h5', compile=False)

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
