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
from plotting import plot_prediction_accuracy
from InjectionFunctions import multiply_variables

if __name__ == '__main__':

    # Training data size
    N = 10000

    # Number of timesteps to skip when creating training data
    sparse = 1

    # Generate some training- and test-data
    raw_data = tsv2arr("eit_test.tsv")
    raw_data = np.array([i for i in raw_data if i[-1] == 0])
    # Remove training batch info and limit training size
    raw_data = raw_data[:N * sparse, :-1]

    x_data, y_data = create_training_data(
        raw_data, SLIDING_WINDOW_LENGTH, sparse)
    # Generate the neural network
    nn = gen_nn(NN_HIDDEN_LAYERS, INJECTION_LAYERS,
                SLIDING_WINDOW_LENGTH, DATA_NUM_VARIABLES, ACTIVATION)

    splitting_index = floor(len(x_data) * (1-VALIDATION_SPLIT))
    x_train = x_data[:splitting_index]
    x_test = x_data[splitting_index:]
    y_train = y_data[:splitting_index]
    y_test = y_data[splitting_index:]

    # Create the injection data
    # TODO: Make general for more than two values
    injection_data = []
    for d in x_train:
        # Just multiplying x and y for now (think this is correct?)
        injection_data.append(
            [d[SLIDING_WINDOW_LENGTH - 1][0] * d[SLIDING_WINDOW_LENGTH - 1][1]])
    injection_data = np.array(injection_data)

    # injection_data = np.zeros((training_data.shape[0], 1))

    # print("Input shape: ", training_data.shape)
    # print("Input data: ")
    # print(training_data[:2, ...])
    # print("Target shape: ", test_data.shape)
    # print("Target data: ")
    # print(test_data[:2, ...])
    # print("Injection shape: ", injection_data.shape)
    # print("injection data: ")
    # print(injection_data[:2, ...])

    # Train the NN
    nn.compile(optimizer=OPTIMIZER, loss=LOSS)
    history = nn.fit(x=[x_train, injection_data], y=[y_train], epochs=10)
    nn.summary()

    plt.plot(history.history['loss'], label='MSE training data')
    # plt.plot(history.history['val_loss'], label='MSE validation data')
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()

    # Then need to predict for the future. Try to see if they match validation data
    starting_point = x_train[-1]
    time_step = abs(x_test[0][0][-1] - x_test[0][1][-1])
    predictions = get_predictions(starting_point, time_step, nn, len(x_test), multiply_variables)

    # This is expected to be perfect, else something is wrong
    t_accuracy = mean_squared_error([i[0][2] for i in x_test], [j[2] for j in predictions])
    print("T-Accuracy with PGML: {}".format(t_accuracy))

    title = "Trained on {} datapoints, window-length {}, time-step {}".format(len(x_train), SLIDING_WINDOW_LENGTH, time_step)
    plot_prediction_accuracy(x_test, predictions, time_step, DATA_NUM_VARIABLES - 1, title)

    # When predictions is obtained, compare with validation data
    x_accuracy = mean_squared_error([i[0][0] for i in x_test], [j[0] for j in predictions])
    print("X-Accuracy with PGML: {}".format(x_accuracy))
    y_accuracy = mean_squared_error([i[0][1] for i in x_test], [j[1] for j in predictions])
    print("Y-Accuracy with PGML: {}".format(y_accuracy))












