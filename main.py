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

    # Then need to predict for the future
    predictions = []
    last_step = training_data[-1]
    time_step = PREDICTION_TIME_STEP
    for i in range(PREDICTION_STEPS):
        injection = last_step[SLIDING_WINDOW_LENGTH-1][0] * last_step[SLIDING_WINDOW_LENGTH-1][1]
        derivative = nn.predict(x=[last_step, injection])
        time_step = PREDICTION_TIME_STEP
        # x(t+1) = x(t) + x´(t) * delta(t)
        new_x = last_step[-1][0] + derivative * time_step
        new_y = last_step[-1][1] + derivative * time_step
        new_time = last_step[-1][-1] + time_step
        # Dont know whats going on the 3rd here? We have index0=x, index1=y. index2=? og index3=timestep?
        new_step = [new_x, new_y, None, new_time]
        last_step = last_step[1:-1] + new_step







