import numpy
from lstmclass import *
from loading_processing_data import *
from training_lstm import *

PARAMETER_OPTIONS_DICT = {
    'learning_rate' : numpy.arange(0.001,0.25,0.003),
    'number_of_neurons' : [i for i in range(16,129,16)],
    'batch_size' : [i for i in range(5,91,15) ],
    'drop_out' : numpy.arange(0.1,0.8,0.05) ,
    # 'activation_functions' : ['tanh','softmax','relu'],
    # 'optimiser' : ['adam', 'rmsprop', 'adagrad']
}

def run_experiments(hyper_parameter_options,iterations, iterations_per_setting, test_x, test_y):

    #dict to catch best settings
    best_options = {
        'learning_rate' : None,
        'number_of_neurons' : None,
        'batch_size' : None,
        'drop_out' : None,
    }

    #overwritten iteratively
    max_acc = -1

    for i in range(iterations):

        learning_rate = hyper_parameter_options['learning_rate'][numpy.random.randint(0,len(hyper_parameter_options['learning_rate']))]

        number_of_neurons = hyper_parameter_options['number_of_neurons'][numpy.random.randint(0,len(hyper_parameter_options['number_of_neurons']))]

        batch_size =  hyper_parameter_options['batch_size'][numpy.random.randint(0,len(hyper_parameter_options['batch_size']))]

        drop_out =  hyper_parameter_options['drop_out'][numpy.random.randint(0,len(hyper_parameter_options['drop_out']))]

        model = model(n_neurons= number_of_neurons, learning_rate = learning_rate, batch_size = batch_size, drop_out = drop_out)

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