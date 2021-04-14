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
    in_train, target_train, in_test, target_test = get_data(DATA_FILE, N, SPARSE, SLIDING_WINDOW_LENGTH)
    
    
    injection_data = create_injection_data(INJECTION_LAYERS, in_train)
    
    inj_runs=[]
    reg_runs=[]
    for i in range(CONFIDENCE_N):
        print("\n"*10)
        print("Run "+str(i+1)+" of " +str(CONFIDENCE_N)+":")
        print("\n")
        # Generate the neural network for injection
        nn_inj = gen_nn(NN_HIDDEN_LAYERS, INJECTION_LAYERS, SLIDING_WINDOW_LENGTH, DATA_NUM_VARIABLES, ACTIVATION)
        nn_reg  = gen_nn(NN_HIDDEN_LAYERS, {}, SLIDING_WINDOW_LENGTH, DATA_NUM_VARIABLES, ACTIVATION)
        
        # Train the NN
        nn_inj.compile(optimizer=OPTIMIZER, loss=LOSS)
        nn_inj.fit(x=get_in_data_for_nn(in_train, injection_data), y=[target_train], epochs=EPOCHS, validation_split=VALIDATION_SPLIT)
        
        nn_reg.compile(optimizer=OPTIMIZER, loss=LOSS)
        nn_reg.fit(x=[in_train], y=[target_train], epochs=EPOCHS, validation_split=VALIDATION_SPLIT)
        
        starting_window = in_test[1]
        testing_time_step = abs(in_test[1][-1][-1] - in_test[0][-1][-1])
        prediction_time_step = testing_time_step/PREDICTION_TIME_STEP_MULTIPLIER
        prediction_steps = min((PREDICTION_MAX_STEPS, len(in_test)))
        testing_t_axis = in_test[:prediction_steps,-1,-1]
        
       
        
        
        inj_predictions = get_predictions(starting_window, prediction_time_step, nn_inj, prediction_steps, INJECTION_LAYERS)
        reg_predictions = get_predictions(starting_window, prediction_time_step, nn_reg, prediction_steps)
        inj_runs.append(inj_predictions)
        reg_runs.append(reg_predictions)
    
    inj_runs=np.array(inj_runs)
    reg_runs=np.array(reg_runs)
    
    plot_confidence_intervals(inj_runs,reg_runs,in_test[SLIDING_WINDOW_LENGTH:prediction_steps+SLIDING_WINDOW_LENGTH,-1,:],CONFIDENCE_STEP,DATA_NUM_VARIABLES-1,CONFIDENCE_PERCENTAGE,labels)








