from InjectionFunctions import *


"""
Structure of data
"""
DATA_NUM_VARIABLES = 3
SLIDING_WINDOW_LENGTH = 5


"""
Neaural Network
"""
LOSS = "mse"
OPTIMIZER = "adam"
VALIDATION_SPLIT = 0.2
PREDICTION_SPLIT = 0.01
NN_HIDDEN_LAYERS = (32, 64, 32)
INJECTION_LAYERS = {1: {"shape": (SLIDING_WINDOW_LENGTH, ), "function": xy}}
ACTIVATION = 'relu'
EPOCHS = 7


"""
Training data
"""
N = 400000  # (Maximum) Training data size
SPARSE = 1  # Number of timesteps to skip when creating training data


"""
Prediction
"""
PREDICTION_TIME_STEP = 0.05


"""
Plotting
"""
WINDOW_LABELS = ["X", "Y", "T"]


