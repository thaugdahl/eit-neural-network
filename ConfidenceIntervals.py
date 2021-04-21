from InjectionPoc import gen_nn, get_predictions
from preprocessing import get_data, split_data
from settings import *
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from math import floor, isnan
from plotting import plot_confidence_intervals
from InjectionFunctions import inject_constant, xy
from utils import get_injection_data, get_target_predictions, split_predictions, split_values, \
    create_injection_data, get_in_data_for_nn
import tensorflow as tf
import datetime



if __name__ == '__main__':
    # Labels
    labels = WINDOW_LABELS


    # Generate some training- and test-data
    
    
    
    runs = []
    for name,inj,layers,window_size in INJECTION_LIST:
        print("\n")
        
        in_train, target_train, in_test, target_test = get_data(DATA_FILE, N, SPARSE, window_size)
    
        starting_window = in_test[1]
        testing_time_step = abs(in_test[1][-1][-1] - in_test[0][-1][-1])
        prediction_time_step = testing_time_step/PREDICTION_TIME_STEP_MULTIPLIER
        prediction_steps = min((PREDICTION_MAX_STEPS, len(in_test)))
        testing_t_axis = in_test[:prediction_steps,-1,-1]
        inj_runs = []
        injection_data = create_injection_data(inj, in_train)
        
        for i in range(CONFIDENCE_N):
        
            print("\n")
            print(name+": Run "+str(i+1)+" of " +str(CONFIDENCE_N)+":")
            print("\n")
            # Generate the neural network for injection
            nn_inj = gen_nn(layers, inj, window_size, DATA_NUM_VARIABLES, ACTIVATION)
           
            # Train the NN
            nn_inj.compile(optimizer=OPTIMIZER, loss=LOSS)
            nn_inj.fit(x=get_in_data_for_nn(in_train, injection_data), y=[target_train], epochs=EPOCHS, validation_split=VALIDATION_SPLIT)
           

            inj_predictions = get_predictions(starting_window, prediction_time_step, nn_inj, prediction_steps, inj)
            
            inj_runs.append(inj_predictions)
            print("\n"*10)
        runs.append(inj_runs)
        
    runs=np.array(runs)
    
    plot_confidence_intervals(runs,in_test[SLIDING_WINDOW_LENGTH:prediction_steps+SLIDING_WINDOW_LENGTH,-1,:],CONFIDENCE_STEP,DATA_NUM_VARIABLES-1,CONFIDENCE_PERCENTAGE,labels,INJECTION_LIST)








