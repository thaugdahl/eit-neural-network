from InjectionPoc import gen_nn
from preprocessing import create_training_data
from solver import generateTrainingData
from settings import *
import numpy as np

if __name__ == '__main__':
    # Generate some training- and test-data
    raw_data = generateTrainingData()
    training_data, test_data = create_training_data(raw_data, SLIDING_WINDOW_LENGTH, DATA_NUM_VARIABLES)
    # Generate the neural network
    nn = gen_nn(NN_HIDDEN_LAYERS, INJECTION_LAYERS, SLIDING_WINDOW_LENGTH, DATA_NUM_VARIABLES)
    # Create the injection data
    injection_data = []
    for d in training_data:
        # Just multiplying x and y for now (think this is correct?)
        injection_data.append([d[SLIDING_WINDOW_LENGTH-1][0] * d[SLIDING_WINDOW_LENGTH-1][1]])
    injection_data = np.array(injection_data)
    # Train the NN
    nn.compile(optimizer=OPTIMIZER, loss=LOSS)
    nn.fit(x=[training_data, injection_data], y=[test_data])
