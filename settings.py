from InjectionFunctions import *


"""
Structure of data
"""
DATA_NUM_VARIABLES = 3
SLIDING_WINDOW_LENGTH = 2


"""
Neaural Network
"""
LOSS = "mse"
OPTIMIZER = "adam"
VALIDATION_SPLIT = 0.2
PREDICTION_SPLIT = 0.05
NN_HIDDEN_LAYERS = (16,32,16)
INJECTION_LAYERS = {3: {"shape": (SLIDING_WINDOW_LENGTH, ), "function": xy}}
ACTIVATION = 'relu'
EPOCHS = 5



"""
Training data
"""
DATA_FILE = 'SparseLV.tsv'
N = 400000  # (Maximum) Training data size
SPARSE = 1  # Number of timesteps to skip when creating training data


"""
Prediction
"""
PREDICTION_TIME_STEP_MULTIPLIER = 2


"""
Plotting
"""
WINDOW_LABELS = ["X", "Y", "T"]


