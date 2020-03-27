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

# check shapes
def check_shapes():
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)
    print('x_validation shape:', x_validation.shape)

    print(y_validation.shape)
    print(y_test.shape)

check_shapes()
