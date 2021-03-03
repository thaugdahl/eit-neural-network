import numpy as np


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
