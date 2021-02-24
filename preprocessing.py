import numpy as np


def createSamplesFromSlidingWindow(data, window_size, skip=0):
    """
    data = [x | y | t | ...]
    returns [sample1 | sample2 | ...]^T, sample_i = [[x_i y_i t_i], [x_i+1 y_i+1 t_i+1]], sample shape (window_size, )
    """

    # Number of samples
    N = data.shape[0] - window_size + 1

    # Index array
    I = np.repeat(np.arange(window_size).reshape((1, window_size)), N, axis=0) + \
        np.arange(N).reshape((N, 1))

    if skip == 0:
        return data[..., :][I]
    else:
        return data[..., :-skip][I]


def createTargets(data, skip=0):
    """
    Create target values using forward differences to approximate derivatives
    """
    if skip == 0:
        derivative = (data[1:, :-1] - data[:-1, :-1]) / \
            (data[1:, -1] - data[:-1, -1]).reshape((data.shape[0] - 1, 1))

    else:
        skip += 1
        derivative = (data[1:, :-skip] - data[:-1, :-skip]) / \
            (data[1:, -skip] - data[:-1, -skip]
             ).reshape((data.shape[0] - 1, 1))

    return derivative


def createTrainingData(data, window_size, skip=0):
    train = createSamplesFromSlidingWindow(data[:-1, ...], window_size, skip)
    target = createTargets(data[window_size - 1:, ...], skip)
    return train, target
