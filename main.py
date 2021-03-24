from InjectionPoc import gen_nn, get_predictions
from preprocessing import get_data, split_data
from settings import *
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from math import floor, isnan
from plotting import plot_prediction_accuracy, plot_derivatives, plot_training_summary, plot_prediction_summary
from InjectionFunctions import multiply_variables, inject_constant, xy
from utils import get_injection_data, get_target_predictions, split_predictions, split_values
import tensorflow as tf
import datetime

# TODO: Make injection more general -
#  should be possible to have several injection functions, as you can have several injection layers
# TODO: Automatic handle NAN-values in predictions (plotting should still work)
if __name__ == '__main__':
    # Labels
    labels = WINDOW_LABELS

    # Set function to be used for injection
    injection_func = xy

    # Generate some training- and test-data
    in_data, target_data = get_data(DATA_FILE, N, SPARSE, SLIDING_WINDOW_LENGTH)
    in_train, in_test, target_train, target_test = split_data(in_data, target_data, PREDICTION_SPLIT)

    # Generate the neural network for injection
    nn_inj = gen_nn(NN_HIDDEN_LAYERS, INJECTION_LAYERS, SLIDING_WINDOW_LENGTH, DATA_NUM_VARIABLES, ACTIVATION)

    # Create callback for tensorboard
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    """
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1)
    """
    
    # Create the injection data
    injection_data = injection_func(in_train)

    # Train the NN
    nn_inj.compile(optimizer=OPTIMIZER, loss=LOSS)
    history = nn_inj.fit(x=[in_train, injection_data], y=[target_train], epochs=EPOCHS, validation_split=VALIDATION_SPLIT)
    plot_training_summary(history, title="Training plot with PGML")

    starting_window = in_train[-1]
    training_time_step = abs(in_test[0][0][-1] - in_test[0][1][-1])
    prediction_time_step = training_time_step/PREDICTION_TIME_STEP_MULTIPLIER

    # Retrieve derivative predictions
    target_predictions = get_target_predictions(in_test, nn_inj, injection_func)
    # Plot derivatives for each variable
    train_t_axis = np.arange(0, in_test[-1][-1][-1] - in_test[0][-1][-1] + training_time_step, training_time_step)
    value_predictions = split_predictions(target_predictions, DATA_NUM_VARIABLES - 1)
    actual_values = split_values(target_test, DATA_NUM_VARIABLES - 1)
    plot_derivatives(train_t_axis, actual_values, value_predictions, title="Derivatives with PGML", labels=WINDOW_LABELS)

    # Then need to predict for the future. Try to see if they match validation data
    predictions = get_predictions(starting_window, prediction_time_step, nn_inj, in_test[-1][-1][-1], injection_func)
    

    # Plot prediction accuracy
    title = "PGML Trained on {} datapoints, window-length {}, time-step {}".format(len(in_train), SLIDING_WINDOW_LENGTH, prediction_time_step)
    plot_prediction_accuracy(in_test, predictions, DATA_NUM_VARIABLES - 1, title, WINDOW_LABELS)

    # Plot the mse between all actual values and all predicted values for the future
    plot_prediction_summary(in_test, predictions, WINDOW_LABELS, "Prediction Summary with PGML")

    # Now repeat the process on a regular neural network without injection to compare
    nn_reg = gen_nn(NN_HIDDEN_LAYERS, {}, SLIDING_WINDOW_LENGTH, DATA_NUM_VARIABLES, ACTIVATION)

    nn_reg.compile(optimizer=OPTIMIZER, loss=LOSS)
    history = nn_reg.fit(x=[in_train], y=[target_train], epochs=EPOCHS, validation_split=VALIDATION_SPLIT)
    plot_training_summary(history, title="Training plot without PGML")

    # Get derivatives and plot for the network
    target_predictions = get_target_predictions(in_test, nn_reg, inj_func=None)
    # Plot derivatives for each variable
    value_predictions = split_predictions(target_predictions, DATA_NUM_VARIABLES - 1)  # Update predictions
    plot_derivatives(train_t_axis, actual_values, value_predictions, title="Derivatives without PGML", labels=WINDOW_LABELS)

    # Then predict for future...
    predictions = get_predictions(starting_window, prediction_time_step, nn_reg, in_test[-1][-1][-1])

    # Plot prediction accuracy
    title = "Trained on {} datapoints, window-length {}, time-step {}".format(
        len(in_train), SLIDING_WINDOW_LENGTH, prediction_time_step)
    plot_prediction_accuracy(in_test, predictions, DATA_NUM_VARIABLES - 1, title, WINDOW_LABELS)

    # Plot the mse between all actual values and all predicted values for the future
    plot_prediction_summary(in_test, predictions, WINDOW_LABELS, "Prediction Summary without PGML")










