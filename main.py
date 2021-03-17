from InjectionPoc import gen_nn, get_predictions
from preprocessing import create_training_data
from solver import tsv2arr
from settings import *
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import matplotlib.pyplot as plt
from math import floor, isnan
from plotting import plot_prediction_accuracy, plot_derivatives
from InjectionFunctions import multiply_variables


# TODO: Sort code into functions
if __name__ == '__main__':

    # Training data size
    N = 400000

    # Number of timesteps to skip when creating training data
    sparse = 1

    # Generate some training- and test-data
    raw_data = tsv2arr("DenseLV.tsv")
    raw_data = np.array([i for i in raw_data if i[-1] == 0])
    # Remove training batch info and limit training size
    raw_data = raw_data[:N * sparse, :-1]

    x_data, y_data = create_training_data(
        raw_data, SLIDING_WINDOW_LENGTH, sparse)
    # Generate the neural network
    nn_inj = gen_nn(NN_HIDDEN_LAYERS, INJECTION_LAYERS,
                SLIDING_WINDOW_LENGTH, DATA_NUM_VARIABLES, ACTIVATION)

    splitting_index = floor(len(x_data) * (1-PREDICTION_SPLIT))
    x_train = x_data[:splitting_index]
    x_prediction = x_data[splitting_index:]
    y_train = y_data[:splitting_index]
    y_prediction = y_data[splitting_index:]

    # Create the injection data
    # TODO: Make general for more than two values
    injection_data = []
    for d in x_train:
        # Just multiplying x and y for now (think this is correct?)
        injection_data.append(
            [d[SLIDING_WINDOW_LENGTH - 1][0] * d[SLIDING_WINDOW_LENGTH - 1][1]])
    injection_data = np.array(injection_data)

    # Train the NN
    nn_inj.compile(optimizer=OPTIMIZER, loss=LOSS)
    history = nn_inj.fit(x=[x_train, injection_data], y=[y_train], epochs=10, validation_split=VALIDATION_SPLIT)
    nn_inj.summary()

    plt.plot(history.history['loss'], label='MSE training data')
    # plt.plot(history.history['val_loss'], label='MSE validation data')
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()

    starting_point = x_train[-1]
    time_step = abs(x_prediction[0][0][-1] - x_prediction[0][1][-1])

    derivative_predictions = []
    for i in range(len(x_prediction)):
        injection = np.array([x_prediction[i][-1][0] * x_prediction[i][-1][1]])
        derivative_predictions.append(nn_inj.predict(x=[np.array([x_prediction[i]]), injection]))

    # TODO: Make general
    x_axis = [time_step * i for i in range(len(x_prediction))]
    x_derivative_predictions = [derivative_predictions[i][0][0] for i in range(len(derivative_predictions))]
    actual_x_derivatives = [y_prediction[i][0] for i in range(len(y_prediction))]
    y_derivative_predictions = [derivative_predictions[i][0][1] for i in range(len(derivative_predictions))]
    actual_y_derivatives = [y_prediction[i][1] for i in range(len(y_prediction))]
    # Plot derivatives
    plot_derivatives(x_axis, [actual_x_derivatives, actual_y_derivatives], [x_derivative_predictions, y_derivative_predictions])

    # Then need to predict for the future. Try to see if they match validation data
    predictions = get_predictions(starting_point, time_step, nn_inj, len(x_prediction), multiply_variables)

    # This is expected to be perfect, else something is wrong
    t_accuracy = mean_squared_error([i[0][2] for i in x_prediction], [j[2] for j in predictions])
    print("T-Accuracy with PGML: {}".format(t_accuracy))

    title = "Trained on {} datapoints, window-length {}, time-step {}".format(len(x_train), SLIDING_WINDOW_LENGTH, time_step)
    plot_prediction_accuracy(x_prediction, predictions, time_step, DATA_NUM_VARIABLES - 1, title)

    # When predictions is obtained, compare with validation data
    x_accuracy = mean_squared_error([i[0][0] for i in x_prediction], [j[0] for j in predictions])
    print("X-Accuracy with PGML: {}".format(x_accuracy))
    y_accuracy = mean_squared_error([i[0][1] for i in x_prediction], [j[1] for j in predictions])
    print("Y-Accuracy with PGML: {}".format(y_accuracy))

    # Now repeat the process on a regular neural network without injection to compare
    nn_reg = gen_nn(NN_HIDDEN_LAYERS, {}, SLIDING_WINDOW_LENGTH, DATA_NUM_VARIABLES, ACTIVATION)

    # Get derivatives and plot for the network
    derivative_predictions_reg = []
    for i in range(len(x_prediction)):
        derivative_predictions_reg.append(nn_reg.predict(x=[np.array([x_prediction[i]])]))

    # TODO: Make general
    x_axis = [time_step * i for i in range(len(x_prediction))]
    x_derivative_predictions_reg = [derivative_predictions_reg[i][0][0] for i in range(len(derivative_predictions_reg))]
    y_derivative_predictions_reg = [derivative_predictions_reg[i][0][1] for i in range(len(derivative_predictions_reg))]
    # Plot derivatives
    plot_derivatives(x_axis, [actual_x_derivatives, actual_y_derivatives],
                     [x_derivative_predictions_reg, y_derivative_predictions_reg])

    # Then predict for future...
    predictions = get_predictions(starting_point, time_step, nn_reg, len(x_prediction), multiply_variables)















