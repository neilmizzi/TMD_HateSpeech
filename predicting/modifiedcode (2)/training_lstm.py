from sklearn.model_selection import train_test_split
from loading_processing_data import *
from lstmclass import *
from keras.preprocessing import sequence

#preparing processed data

#split test and train
x_train, x_test, y_train, y_test = train_test_split(integer_mapped_tweets, TRAIN_LABELS, test_size=0.2, random_state=42)

#split train data again for validation data
x_train, x_validation, y_train, y_validation = train_test_split(x_train,y_train,test_size=0.2, random_state=42)

maxlen = 280
#using keras padding
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
x_validation = sequence.pad_sequences(x_validation,maxlen=maxlen)

# remove to run on full data, used to quickly check outcomes
x_train = x_train[0:100]
x_test = x_test[0:10]
y_train = y_train[0:100]
y_test = y_test[0:10]

x_validation = x_validation[0:10]
y_validation = y_validation[0:10]

# check shapes

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print('x_validation shape:', x_validation.shape)

print(y_validation.shape)
print(y_test.shape)

# call model
lstm = LSTM_model(maxlen)
#intiliase
lstm.initialise_model()
#train
lstm.train_model(x_train,y_train,x_validation,y_validation)
#evaluate
lstm.evaluate_model(x_test,y_test)
#compile

#transform
prediction_data = ['this is dummy data','yes it is'] #paste here your prediction data as a list of strings

lstm.transform_prediction_data(prediction_data)

#get predictions
scores, labels = lstm.predict_scores_and_labels()

print(scores,labels)
