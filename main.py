from InjectionPoc import gen_nn
from preprocessing import create_training_data
from solver import tsv2arr
from settings import *
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # Training data size
    N = 1000

    # Number of timesteps to skip when creating training data
    sparse = 10

    # Generate some training- and test-data
    raw_data = tsv2arr("eit_test.tsv")

    # Remove training batch info and limit training size
    raw_data = raw_data[:N * sparse, :-1]

    training_data, test_data = create_training_data(
        raw_data, SLIDING_WINDOW_LENGTH, sparse)
    # Generate the neural network
    nn = gen_nn(NN_HIDDEN_LAYERS, INJECTION_LAYERS,
                SLIDING_WINDOW_LENGTH, DATA_NUM_VARIABLES)

    # Create the injection data
    injection_data = []
    for d in training_data:
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
    history = nn.fit(x=[training_data, injection_data], y=[
        test_data], epochs=100, validation_split=0.5)

    plt.plot(history.history['loss'], label='MSE training data')
    plt.plot(history.history['val_loss'], label='MSE validation data')
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()

    # # Then need to predict for the future
    # predictions = []
    # last_step = training_data[-1]
    # time_step = PREDICTION_TIME_STEP
    # for i in range(PREDICTION_STEPS):
    #     injection = last_step[SLIDING_WINDOW_LENGTH -
    #                           1][0] * last_step[SLIDING_WINDOW_LENGTH - 1][1]
    #     derivative = nn.predict(x=[last_step, injection])
    #     time_step = PREDICTION_TIME_STEP
    #     # x(t+1) = x(t) + x´(t) * delta(t)
    #     new_x = last_step[-1][0] + derivative * time_step
    #     new_y = last_step[-1][1] + derivative * time_step
    #     new_time = last_step[-1][-1] + time_step
    #     # Dont know whats going on the 3rd here? We have index0=x, index1=y. index2=? og index3=timestep?
    #     new_step = [new_x, new_y, None, new_time]
    #     last_step = last_step[1:-1] + new_step
