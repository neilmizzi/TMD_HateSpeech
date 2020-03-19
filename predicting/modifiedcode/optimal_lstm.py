from lstmclass import *
from loading_processing_data import *
from training_lstm import *

optimal_options = {'learning_rate': 0.004, 'number_of_neurons': 80, 'batch_size': 15, 'drop_out': 0.1}

lstm = LSTM_model(280,n_neurons=optimal_options['number_of_neurons'],learning_rate=optimal_options['learning_rate'],
                  batch_size=optimal_options['batch_size'],drop_out= optimal_options['drop_out'], bidirectional= True)

lstm.initialise_model()
lstm.train_model(x_train,y_train,x_validation,y_validation)
lstm.evaluate_model(x_test,y_test)
