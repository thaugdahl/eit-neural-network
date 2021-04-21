from InjectionFunctions import *


"""
Structure of data
"""
DATA_NUM_VARIABLES = 3
SLIDING_WINDOW_LENGTH = 7

"""
Neaural Network
"""
LOSS = "mse"
OPTIMIZER = "adam"
LEARNING_RATE = 0.00005
VALIDATION_SPLIT = 0.2
NN_HIDDEN_LAYERS = (32,64,32)
INJECTION_LAYERS = {}
"""
{1: {"shape": (SLIDING_WINDOW_LENGTH, ), "function": xy},
                    1: {"shape": (SLIDING_WINDOW_LENGTH, ), "function": xz}}
"""

ACTIVATION = 'relu'
EPOCHS = 10



"""
Training data
"""
DATA_FILE = 'LargeDuffing.tsv'
N = 400000  # (Maximum) Training data size
SPARSE = 1 # Number of timesteps to skip when creating training data


"""
Prediction
"""
PREDICTION_TIME_STEP_MULTIPLIER = 1
PREDICTION_MAX_STEPS = 500


"""
Plotting
"""
WINDOW_LABELS = ["X", "Y", "T"]

"""
Statistics
"""
CONFIDENCE_PERCENTAGE = 95
CONFIDENCE_STEP = 1
CONFIDENCE_N = 1

INJECTION_LIST = [
        ["Large",{},(64,128,128,128,64),7],
        ["Medium",{},(32,64,64,32),7],
        ["Small",{},(16,32,16),7]]


