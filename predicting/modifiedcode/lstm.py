from __future__ import print_function
from sklearn.model_selection import train_test_split
import pandas as pd
from onehotencoding import *
from keras.preprocessing import sequence
import itertools
from keras.callbacks import EarlyStopping
import h5py
import numpy
# from hyperparametertuning import *


# training and evaluation data
TRAIN_DATA = pd.read_csv('/Users/b2077/PycharmProjects/pleasework/TMD_HateSpeech/data sets/Data-Set 33_33_34.csv',sep = '\t')
TRAIN_TWEETS = TRAIN_DATA['0'].astype(str).values # training data
TRAIN_LABELS = TRAIN_DATA['1'].values
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

maxlen = 280
#using keras padding
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


#max(set(unique_characters)) + 1



PARAMETER_OPTIONS_DICT = {
    'learning_rate' : numpy.arange(0.001,0.25,0.003),
    'number_of_neurons' : [i for i in range(16,129,16)],
    'batch_size' : [i for i in range(5,91,15) ],
    'drop_out' : numpy.arange(0.1,0.8,0.05) ,
    # 'activation_functions' : ['tanh','softmax','relu'],
    # 'optimiser' : ['adam', 'rmsprop', 'adagrad']
}

# print(PARAMETER_OPTIONS_DICT['learning_rate'][5])

def run_experiments(hyper_parameter_options,iterations, iterations_per_setting, test_x, test_y):


    best_options = {
        'learning_rate' : None,
        'number_of_neurons' : None,
        'batch_size' : None,
        'drop_out' : None,
    }

    max_acc = -9999

    for i in range(iterations):

        learning_rate = hyper_parameter_options['learning_rate'][numpy.random.randint(0,len(hyper_parameter_options['learning_rate']))]

        number_of_neurons = hyper_parameter_options['number_of_neurons'][numpy.random.randint(0,len(hyper_parameter_options['number_of_neurons']))]

        batch_size =  hyper_parameter_options['batch_size'][numpy.random.randint(0,len(hyper_parameter_options['batch_size']))]

        drop_out =  hyper_parameter_options['drop_out'][numpy.random.randint(0,len(hyper_parameter_options['drop_out']))]

        model = LSTM_model(n_neurons= number_of_neurons, learning_rate = learning_rate, batch_size = batch_size, drop_out = drop_out)

        accuracy = []

        model.initialise_model()
        iterations = 1
        for i in range(iterations_per_setting): # statistical significance?
            model.train_model()
            results = model.model.evaluate(test_x,test_y, verbose=1)
            acc = results[1]
            accuracy.append(acc)

        avg_accuracy = sum(accuracy) / len(iterations_per_setting)

        if avg_accuracy > max_acc:
            max_acc = avg_accuracy
            best_options['learning_rate'] = learning_rate
            best_options['number_of_neurons'] = number_of_neurons
            best_options['batch_size'] = batch_size
            best_options['drop_out'] = drop_out

    return max_acc, best_options


max_acc, best_options = run_experiments(PARAMETER_OPTIONS_DICT,10,x_test,y_test)

print(max_acc)

print(best_options)

#
# test_model = LSTM_model()
# test_model.initialise_model()
# test_model.transform_prediction_data(TRAIN_TWEETS[500:550])
#
# predictions = test_model.predict_scraped_data_labels()


