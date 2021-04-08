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
NN_HIDDEN_LAYERS = (16,32,16)
INJECTION_LAYERS = {3: {"shape": (SLIDING_WINDOW_LENGTH, ), "function": cos_t(0.4)},
                    1: {"shape": (SLIDING_WINDOW_LENGTH, ), "function": x_cubed}}
ACTIVATION = 'relu'
EPOCHS = 5



"""
Training data
"""
DATA_FILE = 'Duffing.tsv'
N = 400000  # (Maximum) Training data size
SPARSE = 1 # Number of timesteps to skip when creating training data


"""
Prediction
"""
PREDICTION_TIME_STEP_MULTIPLIER = 1
PREDICTION_MAX_STEPS = 100


"""
Plotting
"""
WINDOW_LABELS = ["X", "Y", "T"]

"""
Statistics
"""
CONFIDENCE_PERCENTAGE = 95
CONFIDENCE_STEP = 1
CONFIDENCE_N = 3


