import numpy as np


def createSamplesFromSlidingWindow(data, window_size, remove=0):
    """
    data = [x | y | t | ...]
    returns [sample1 | sample2 | ...]^T, sample_i = [[x_i y_i t_i], [x_i+1 y_i+1 t_i+1]], sample shape (window_size, )
    """

    # Number of samples
    N = data.shape[0] - window_size + 1

    # Index array
    I = np.repeat(np.arange(window_size).reshape((1, window_size)), N, axis=0) + \
        np.arange(N).reshape((N, 1))

    if remove == 0:
        return data[..., :][I]
    else:
        return data[..., :-remove][I]


def createTargets(data, remove=0):
    """
    Create target values using forward differences to approximate derivatives
    """
    if remove == 0:
        derivative = (data[1:, :-1] - data[:-1, :-1]) / \
            (data[1:, -1] - data[:-1, -1]).reshape((data.shape[0] - 1, 1))

    else:
        remove += 1
        derivative = (data[1:, :-remove] - data[:-1, :-remove]) / \
            (data[1:, -remove] - data[:-1, -remove]
             ).reshape((data.shape[0] - 1, 1))

    return derivative


def createTrainingData(data, window_size, skip=1, remove=0):
    data = data[::skip, ...]
    train = createSamplesFromSlidingWindow(data[:-1, ...], window_size, remove)
    target = createTargets(data[window_size - 1:, ...], remove)
    return train, target
