from lstmclass import *
from loading_processing_data import *
from training_lstm import *

#test
optimal_options = {'learning_rate': 0.004, 'number_of_neurons': 80, 'batch_size': 15, 'drop_out': 0.1}

lstm = LSTM_model(280,n_neurons=optimal_options['number_of_neurons'],learning_rate=optimal_options['learning_rate'],
                  batch_size=optimal_options['batch_size'],drop_out= optimal_options['drop_out'], bidirectional= True) # adjust as you see fit


lstm.initialise_model()
lstm.train_model(x_train,y_train,x_validation,y_validation)


twintoutput = pd.read_csv('/Users/b2077/PycharmProjects/pleasework/TMD_HateSpeech/predicting/modifiedcode (2)/datafiles/evaluation_set.csv')
TWEETS = list(twintoutput['tweets'].astype(str).values)

lstm.evaluate_model(x_test,y_test) # gives you an accuracy
lstm.transform_prediction_data(TWEETS) # you need to put your raw text as a list in here

labels = lstm.test_set_label_predictions(x_test) # change what you want to predict

prediction_dict = { 'Predictions': labels, "True Labels": y_test # change as you like


}

predictions_datafame = pd.DataFrame(prediction_dict)
predictions_datafame.to_csv("optimalsettingsXTEST_",index = False)





