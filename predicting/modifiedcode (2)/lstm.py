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

# training and evaluation data
TRAIN_DATA = pd.read_csv('labeled_data (1).txt')
TRAIN_TWEETS = TRAIN_DATA['tweet'].astype(str).values # training data
TRAIN_LABELS = TRAIN_DATA['class'].values
TRAIN_UNIQUE_LABELS = set(TRAIN_LABELS)

#predict data - replace first none with pd.read_csv(filepath), second none with column name
PREDICT_DATA, TWEET_COLUMN = None, None
#PREDICT_TWEETS = PREDICT_DATA[TWEET_COLUMN].astype(str).values #comment out when filled in prediction data

# integer mapping for the train data
integer_mapped_tweets = [map_to_ints(t) for t in TRAIN_TWEETS]
unique_characters = merged = list(itertools.chain.from_iterable(integer_mapped_tweets)) #flattening nested list: making one list out of all the lists so we can use set()

# check how many letters their are in the tweets
print("Unique characters:",len(set(unique_characters)))

#split test and train
x_train, x_test, y_train, y_test = train_test_split(integer_mapped_tweets, TRAIN_LABELS, test_size=0.2, random_state=42)

#split train data again for validation data
x_train, x_validation, y_train, y_validation = train_test_split(x_train,y_train,test_size=0.2, random_state=42)

maxlen = 280     # max length a tweet can be, this can be tweaked for better performance maybe
# NOTE: using keras padding
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
x_validation = sequence.pad_sequences(x_validation,maxlen=maxlen)

# remove to run on full data
x_train = x_train[0:100]
x_test = x_test[0:10]
y_train = y_train[0:100]
y_test = y_test[0:10]

x_validation = x_validation[0:10]
y_validation = y_validation[0:10]

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print('x_validation shape:', x_validation.shape)

print(y_validation.shape)
print(y_test.shape)


class LSTM_model:
    def __init__(self,n_neurons = 50 ,batch_size = 32,epochs = 5,max_features = max(set(unique_characters)) + 1 ,input_length = 280,output_dim =128, bidirectional = True, \
                 drop_out = 0.5,num_classes = 3, activation_function = 'softmax', loss_function = 'sparse_categorical_crossentropy', \
                 optimiser = 'rmsprop'):
        self.n_neurons = n_neurons
        self.batch_size = batch_size
        self.epochs = epochs
        self.max_features = max_features
        # self.input_dim = max_features
        self.input_length = input_length
        self.output_dim = output_dim
        self.biderectional = bidirectional
        self.drop_out = drop_out
        self.num_classes = num_classes
        self.activation_function = activation_function
        self.loss_function = loss_function
        self.optimiser = optimiser
        self.model = Sequential()


    def initialise_model(self):
        self.model.add(Embedding(input_dim=self.max_features, input_length= self.input_length, output_dim=self.output_dim))
        if self.biderectional == False:
            self.model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
            self.model.add(LSTM(self.num_neurons,return_sequences=True,input_shape=(MAX_LENGTH,len(CHARS)))) #MAXLENGTH AND CHARS are constants from the onehotencoding.py
            self.model.add(LSTM(self.num_neurons,return_sequences=True)) #MAXLENGTH AND CHARS are constants from the onehotencoding.py
            self.model.add(Dropout(.2))
            self.model.add(Flatten())
        else:
            self.model.add(Bidirectional(LSTM(64)))
            self.model.add(Dropout(0.5))


        self.model.add(Dense(self.num_classes, activation=self.activation_function)) # 3 refers to the number of categories
        self.model.compile(self.optimiser, self.loss_function, metrics=['accuracy'])

        self.model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.epochs, validation_data=(x_validation, y_validation))

        self.model.summary()


    def evaluate_model(self,test_x,test_y):
        self.scores = self.model.evaluate(test_x,test_y, verbose=1)
        print('Test loss:', self.scores[0])
        print('Test accuracy:', self.scores[1])

    def transform_prediction_data(self,prediction_data: list):
        self.prediction_data = [map_to_ints(t) for t in prediction_data]
        self.prediction_data = sequence.pad_sequences(self.prediction_data, maxlen=maxlen)

    def predict_scraped_data_labels(self):
        predictions = self.model.predict_classes(self.prediction_data)
        return predictions

    def predict_scraped_data_scores(self):
        scores = self.model.predict(self.prediction_data)
        return scores

