from __future__ import print_function
from onehotencoding import *
from keras.models import Sequential
from keras.layers import Dense, Dropout,Flatten,LSTM, Embedding, Bidirectional
import numpy as np
import keras
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
import h5py


class LSTM_model:
    def __init__(self, max_features,n_neurons=50, batch_size=32,
                 epochs=5,
                 input_length=280, output_dim=128, bidirectional=True,
                 drop_out=0.2, num_classes=3, activation_function='softmax',
                 loss_function='sparse_categorical_crossentropy',
                 learning_rate=0.001, maxlen = 280
                 ):
        self.n_neurons = n_neurons
        self.batch_size = batch_size
        self.epochs = epochs
        self.max_features = max_features
        # self.input_dim = max_features
        self.input_length = input_length
        self.output_dim = output_dim
        self.biderectional = bidirectional
        self.drop_out = drop_out
        self.maxlen = maxlen
        self.num_classes = num_classes
        self.activation_function = activation_function
        self.loss_function = loss_function
        self.optimiser = keras.optimizers.rmsprop(learning_rate=learning_rate)
        self.model = Sequential()
        self.early_stopping = EarlyStopping(monitor='val_loss', mode=min, verbose=1, min_delta=0.05)
        # self.model_check_point = ModelCheckpoint()

    def initialise_model(self):
        self.model.add(
            Embedding(input_dim=self.max_features, input_length=self.input_length, output_dim=self.output_dim))

        if self.biderectional == False:
            self.model.add(LSTM(128, dropout=self.drop_out, recurrent_dropout=self.drop_out))
            self. model.add(Dense(3, activation='softmax'))  # 3 refers to the number of categories

        else:
            self.model.add(Bidirectional(LSTM(self.n_neurons)))
            self.model.add(Dropout(self.drop_out))

        self.model.add(
            Dense(self.num_classes, activation=self.activation_function))  # 3 refers to the number of categories
        self.model.compile(self.optimiser, self.loss_function, metrics=['accuracy'])
        self.model.summary()

    def train_model(self,x_train,y_train,x_validation, y_validation):
        if self.biderectional == True:
            self.model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.epochs,
                           validation_data=(x_validation, y_validation),
                           callbacks=[self.early_stopping])
        else:
            self.model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.epochs,
                           validation_data=(x_validation, y_validation),
                           )


    def evaluate_model(self, test_x, test_y):
        self.scores = self.model.evaluate(test_x, test_y, verbose=1)
        print('Test loss:', self.scores[0])
        print('Test accuracy:', self.scores[1])

    def transform_prediction_data(self, prediction_data: list):
        self.prediction_data = [map_to_ints(t) for t in prediction_data]
        self.prediction_data = sequence.pad_sequences(self.prediction_data, maxlen=self.maxlen)

    def predict_scraped_data_labels_scores(self):
        labels = self.model.predict_classes(self.prediction_data)
        return labels

    def test_set_label_predictions(self,data):
        labels = self.model.predict_classes(data)
        return labels

    def predict_scraped_data_scores(self):
        scores = self.model.predict(self.prediction_data)
        return scores

    def predict_scores_and_labels(self):
        labels = self.model.predict_classes(self.prediction_data)
        scores = self.model.predict(self.prediction_data)
        return labels, scores