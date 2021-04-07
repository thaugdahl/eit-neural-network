import numpy as np
from solver import tsv2arr
from math import floor
from settings import *


def create_sliding_window(data: np.ndarray, window_size: int):
    """
    Converts data to sliding window data
    :param data: Timeseries data to convert to sliding window data. Columns of data are x1,..,xn,t.
    :param window_size: Length of sliding window.
    :return: Sliding window data. Each row contains all the data in one sliding window.
    """

    # Number of sliding windows to create
    N = data.shape[0] - window_size + 1

    # Index array
    I = np.repeat(np.arange(window_size).reshape((1, window_size)), N, axis=0) + \
        np.arange(N).reshape((N, 1))

    return data[..., :][I]


def approximate_derivatives(data: np.ndarray):
    """
    Aapproximate derivative using forward difference.
    :param data: Timeseries data with timestamps in the last column.
    :return: Numpy array with derivatives.
    """

    derivative = (data[1:, :-1] - data[:-1, :-1]) / \
        (data[1:, -1] - data[:-1, -1]).reshape((data.shape[0] - 1, 1))

    return derivative


def create_training_data(data: np.ndarray, window_size: int, step: int):
    """
    Create training data of sliding windows with target values as derivatives.
    :param data: Timeseries data with timestamps in the last column.
    :param window_size: Length of sliding window.
    :param step: For sparsing the data. Step 1 uses all the data, step 2 uses every other datapoint and so on
    :return: Tuple with train and target data
    """

    sparse_data = data[::step, ...]

    train_data = create_sliding_window(sparse_data[:-1, ...], window_size)
    target_data = approximate_derivatives(sparse_data[window_size - 1:, ...])

    return train_data, target_data


def get_data(file, n, sparse, window_size):
    """
    Returns data from the file as a numpy array.
    :param file: The file to read data from
    :param n: Number of rows to include (max) from file
    :param sparse: If sparse=x, only use every x-th row
    :param window_size: The size of the sliding window for data
    :return: The data from file
    """
    # Generate some training- and test-data
    raw_data = tsv2arr(file)
    in_train=[]
    target_train = []
    for i in range(0,int(raw_data[-1,-1])):   
        run = np.array(list(filter(lambda x:x[-1]==i,raw_data)))
        run = create_training_data(run[:,:-1], window_size,sparse)
        in_train.append(run[0])
        target_train.append(run[1])
    in_train = np.array(in_train)    
    in_train = np.reshape(in_train,(-1,window_size,DATA_NUM_VARIABLES))    
    
    target_train = np.array(target_train)
    target_train = np.reshape(target_train,(-1,DATA_NUM_VARIABLES-1))
        
    last_run = np.array(list(filter(lambda x:x[-1]==raw_data[-1,-1], raw_data)))
    in_test,target_test = create_training_data(last_run[:,:-1],window_size,sparse)
    return in_train, target_train, in_test, target_test


def split_data(x_data, y_data, split):
    """
    Splits the training (x_data) and testing (y_data) data into four parts.
    The first part is the x-data for training, second is x-data for prediction (backprop to nn),
    third part is y-data for training and last part is y-data for prediction
    :param x_data: General input data
    :param y_data: General output data
    :param split: The split percentage (how large should the prediction data be in percentage)
    :return: x-data for training, x-data for prediction, y-data for training, y-data for prediction
    """
    splitting_index = floor(len(x_data) * (1 - split))
    x_train = x_data[:splitting_index]
    x_prediction = x_data[splitting_index:]
    y_train = y_data[:splitting_index]
    y_prediction = y_data[splitting_index:]
    return x_train, x_prediction, y_train, y_prediction
