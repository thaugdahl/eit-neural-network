from InjectionPoc import gen_nn, get_predictions
from preprocessing import get_data, split_data
from settings import *
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from math import floor, isnan
from plotting import plot_prediction_accuracy, plot_derivatives, plot_training_summary, plot_prediction_summary
from InjectionFunctions import inject_constant, xy
from utils import get_injection_data, get_target_predictions, split_predictions, split_values, \
    create_injection_data, get_in_data_for_nn
import tensorflow as tf
import datetime

# TODO: Automatic handle NAN-values in predictions (plotting should still work)
if __name__ == '__main__':
    # Labels
    labels = WINDOW_LABELS

    # Set function to be used for injection
    injection_func = xy

    # Generate some training- and test-data
    in_train, target_train, in_test, target_test = get_data(DATA_FILE, N, SPARSE, SLIDING_WINDOW_LENGTH)


    # Generate the neural network for injection
    nn_inj = gen_nn(NN_HIDDEN_LAYERS, INJECTION_LAYERS, SLIDING_WINDOW_LENGTH, DATA_NUM_VARIABLES, ACTIVATION)
    """
    # Create callback for tensorboard
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1)
    """

    # Create the injection data
    injection_data = create_injection_data(INJECTION_LAYERS, in_train)
    
    # Train the NN
    nn_inj.compile(optimizer=OPTIMIZER, loss=LOSS)

    history = nn_inj.fit(x=get_in_data_for_nn(in_train, injection_data), y=[target_train], epochs=EPOCHS, validation_split=VALIDATION_SPLIT)
    plot_training_summary(history, title="Training plot with PGML")
    
    starting_window = in_test[1]
    testing_time_step = abs(in_test[1][-1][-1] - in_test[0][-1][-1])
    prediction_time_step = testing_time_step/PREDICTION_TIME_STEP_MULTIPLIER
    prediction_steps = min((PREDICTION_MAX_STEPS, len(in_test)))
    testing_t_axis = in_test[:prediction_steps,-1,-1]
    
    # Retrieve derivative predictions
    target_predictions = get_target_predictions(in_test[:prediction_steps], nn_inj, INJECTION_LAYERS)[:,0,:]
    # Plot derivatives for each variable

    plot_derivatives(testing_t_axis,target_test[:prediction_steps], target_predictions, title="Derivatives with PGML", labels=WINDOW_LABELS)

    # Then need to predict for the future. Try to see if they match validation data
    predictions = get_predictions(starting_window, prediction_time_step, nn_inj, prediction_steps, INJECTION_LAYERS)
    

    # Plot prediction accuracy
    title = "PGML Trained on {} datapoints, window-length {}, time-step {}".format(len(in_train), SLIDING_WINDOW_LENGTH, prediction_time_step)
    plot_prediction_accuracy(in_test[:prediction_steps,:,:], predictions, DATA_NUM_VARIABLES - 1, title, WINDOW_LABELS)

    # Plot the mse between all actual values and all predicted values for the future
    plot_prediction_summary(in_test[:prediction_steps,:,:], predictions, WINDOW_LABELS, "Prediction Summary with PGML")

    # Now repeat the process on a regular neural network without injection to compare
    nn_reg = gen_nn(NN_HIDDEN_LAYERS, {}, SLIDING_WINDOW_LENGTH, DATA_NUM_VARIABLES, ACTIVATION)

    nn_reg.compile(optimizer=OPTIMIZER, loss=LOSS)
    history = nn_reg.fit(x=[in_train], y=[target_train], epochs=EPOCHS, validation_split=VALIDATION_SPLIT)
    plot_training_summary(history, title="Training plot without PGML")

    # Get derivatives and plot for the network
    target_predictions = get_target_predictions(in_test[:prediction_steps,:,:], nn_reg)[:,0,:]
    # Plot derivatives for each variable

    plot_derivatives(testing_t_axis,target_test[:prediction_steps], target_predictions, title="Derivatives without PGML", labels=WINDOW_LABELS)

    # Then predict for future...
    predictions = get_predictions(starting_window, prediction_time_step, nn_reg, prediction_steps)

    # Plot prediction accuracy
    title = "Trained on {} datapoints, window-length {}, time-step {}".format(
        len(in_train), SLIDING_WINDOW_LENGTH, prediction_time_step)
    plot_prediction_accuracy(in_test[:prediction_steps], predictions, DATA_NUM_VARIABLES - 1, title, WINDOW_LABELS)

    # Plot the mse between all actual values and all predicted values for the future
    plot_prediction_summary(in_test[:prediction_steps], predictions, WINDOW_LABELS, "Prediction Summary without PGML")








