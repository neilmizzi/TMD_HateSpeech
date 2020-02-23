from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import SpatialDropout1D
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from onehotencoding import *
from keras.callbacks import EarlyStopping
from keras.layers.core import Dense, Activation, Dropout
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM

DATA = pd.read_csv('/Users/b2077/PycharmProjects/Maincodes/TDM/labeled_data (1).txt')
TWEETS = DATA['tweet'].astype(str).values
LABELS = DATA['class'].values
UNIQUE_LABELS = set(LABELS)

#encode and pad tweets, labels are already coded in this data
padded_encoded_tweets = [pad_tweet(get_one_hot(tweet)) for tweet in TWEETS]


#split test and train
x_train, x_test, y_train, y_test = train_test_split( padded_encoded_tweets, LABELS, test_size=0.2, random_state=42)

#reshape for timesteps
x_train = np.array(x_train)
x_test = np.array(x_test)

print(x_train.shape)

x_train = np.reshape(x_train.shape[0],1,x_train.shape[1])
x_test = np.reshape(x_test.shape[0],1,x_test.shape[1])





#initialise hidden nodes, based on rule of thumb can be improved
hidden_nodes = int(2/3 * (MAX_LENGTH * len(CHARS)))

#creating model
model = Sequential()
model.add(LSTM(50, return_sequences=False,input_shape= padded_encoded_tweets[0].shape,))
model.add(Dropout(0.2))
model.add(Dense(units=len(UNIQUE_LABELS)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
batch_size=1000
model.fit(x_train, y_train, batch_size=batch_size, epochs=10)


