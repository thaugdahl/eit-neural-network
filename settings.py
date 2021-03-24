LOSS = "mse"
OPTIMIZER = "adam"
VALIDATION_SPLIT = 0.2
PREDICTION_SPLIT = 0.01
NN_HIDDEN_LAYERS = (32, 64, 32)
INJECTION_LAYERS = {1: (1, )}
ACTIVATION = 'relu'
N = 400000  # (Maximum) Training data size
SPARSE = 3  # Number of timesteps to skip when creating training data
PREDICTION_TIME_STEP = 0.05
WINDOW_LABELS = ["X", "Y", "T"]

"""
Structure of data
"""
DATA_NUM_VARIABLES = 3
SLIDING_WINDOW_LENGTH = 5
