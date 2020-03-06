from __future__ import print_function
from sklearn.model_selection import train_test_split
import pandas as pd
from onehotencoding import *
from keras.models import Sequential
from keras.layers import Dense, Dropout,Flatten,LSTM, Embedding, Bidirectional
import numpy as np
from keras.datasets import imdb
from keras.preprocessing import sequence
import itertools
from keras.callbacks import EarlyStopping

DATA = pd.read_csv('labeled_data (1).txt')
TWEETS = DATA['tweet'].astype(str).values
LABELS = DATA['class'].values
UNIQUE_LABELS = set(LABELS)

#encode and pad tweets, labels are already coded in this data because they are ints
#padded_encoded_tweets = [pad_tweet(get_one_hot(tweet)) for tweet in TWEETS] # this returns a list of all the tweets
                                                                            # in the form of a 280 * 100 matrix
# NOTE: I removed the one-hot encoding, simply mapping to integers
# we need to look further into this
integer_mapped_tweets = [map_to_ints(t) for t in TWEETS]

XX = merged = list(itertools.chain.from_iterable(integer_mapped_tweets)) #flattening nested list: making one list out of all the lists so we can use set()

# check how many letters their are in the tweets
print(set(XX))

#split test and train
x_train, x_test, y_train, y_test = train_test_split(integer_mapped_tweets, LABELS, test_size=0.2, random_state=42)

#split train data again for validation data
x_train, x_validation, y_train, y_validation = train_test_split(x_train,y_train,test_size=0.2, random_state=42)

maxlen = 280     # max length a tweet can be, this can be tweaked for better performance maybe
# NOTE: using keras padding
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
x_validation = sequence.pad_sequences(x_validation,maxlen=maxlen)

# remove to run on full data
# x_train = x_train[0:100]
# x_test = x_test[0:10]
# y_train = y_train[0:100]
# y_test = y_test[0:10]
#
# x_validation = x_validation[0:10]
# y_validation = y_validation[0:10]

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print('x_validation shape:', x_validation.shape)

print(y_validation.shape)
print(y_test.shape)


#setting up a model parameters
num_neurons = 50 # arbitrary number needs to
batch_size = 32 # also kinda arbitrary
epochs = 5 # arbitrary, we should lookinto auto stopping epochs
max_features =  max(set(XX)) + 1 #len(set(XX))    # highest integer map number will give us the max len + 1 does not work because it skips a number in integer coding

#model itself
model = Sequential()
# copied the next two lines from Keras's sentiment-LSTM tutorial
model.add(Embedding(input_dim=max_features, input_length= 280, output_dim=128))
# model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))

#making it bi-directional
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.5))


#model.add(LSTM(num_neurons,return_sequences=True,input_shape=(MAX_LENGTH,len(CHARS)))) #MAXLENGTH AND CHARS are constants from the onehotencoding.py
#model.add(LSTM(num_neurons,return_sequences=True)) #MAXLENGTH AND CHARS are constants from the onehotencoding.py
#model.add(Dropout(.2))
#model.add(Flatten())
model.add(Dense(3, activation='softmax')) # 3 refers to the number of categories

# NOTE: binary_crossentropy is for binary prediction, you should have categorical_crossentropy at least

model.compile('rmsprop','sparse_categorical_crossentropy',metrics=['accuracy'])

model.summary()

model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs, validation_data=(x_validation,y_validation))

scores = model.evaluate(x_test, y_test, verbose=1)

predictions = model.predict_classes(x_test)

print(predictions)

print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
