from __future__ import print_function

from sklearn.model_selection import train_test_split
import pandas as pd
from onehotencoding import *
from keras.models import Sequential
from keras.layers import Dense, Dropout,Flatten,LSTM, Embedding
import numpy as np
from keras.datasets import imdb
from keras.preprocessing import sequence
import itertools


DATA = pd.read_csv('labeled_data (1).txt')
TWEETS = DATA['tweet'].astype(str).values
LABELS = DATA['class'].values
UNIQUE_LABELS = set(LABELS)

#encode and pad tweets, labels are already coded in this data because they are ints
#padded_encoded_tweets = [pad_tweet(get_one_hot(tweet)) for tweet in TWEETS] # this returns a list of all the tweets
                                                                            # in the form of a 280 * 100 matrix

# NOTE: I removed the one-hot encoding, simply mapping to integers
X = [map_to_ints(t) for t in TWEETS]

XX = merged = list(itertools.chain.from_iterable(X)) #flattening nestedlist
print(len(set(XX)))

#split test and train
#x_train, x_test, y_train, y_test = train_test_split(padded_encoded_tweets, LABELS, test_size=0.2, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(X, LABELS, test_size=0.2, random_state=42)


maxlen = 50     # FIXME
# NOTE: using keras padding
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

print(x_train)

# TODO remove these lines when ready to run on full data
# x_train = x_train[0:100]
# x_test = x_test[0:10]
# y_train = y_train[0:100]
# y_test = y_test[0:10]
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

#setting up a model parameters
num_neurons = 50 # arbitrary number needs to
batch_size = 32 # also kinda arbitrary
epochs = 5 # arbitrary, we should lookinto auto stopping epochs
max_features = 280    # FIXME I copied this from Keras's sentiment-LSTM tutorial: https://keras.io/examples/imdb_lstm/
#model itself
model = Sequential()
# copied the next two lines from Keras's sentiment-LSTM tutorial
model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
#model.add(LSTM(num_neurons,return_sequences=True,input_shape=(MAX_LENGTH,len(CHARS)))) #MAXLENGTH AND CHARS are constants from the onehotencoding.py
#model.add(LSTM(num_neurons,return_sequences=True)) #MAXLENGTH AND CHARS are constants from the onehotencoding.py
#model.add(Dropout(.2))
#model.add(Flatten())
model.add(Dense(3, activation='softmax')) # 3 refers to the number of categories

# NOTE: binary_crossentropy is for binary prediction, you should have categorical_crossentropy at least
# This blog explains why using sparse_categorical_crossentropy: https://www.dlology.com/blog/how-to-use-keras-sparse_categorical_crossentropy/
model.compile('rmsprop','sparse_categorical_crossentropy',metrics=['accuracy'])
model.summary()
model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs, validation_data=(x_test,y_test))
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
