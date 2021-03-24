import numpy as np


def get_injection_data(arr, inj_func):
    """
    Returns a list of injection data, obtained by running the injection function on each of the elements in array
    :param arr: The array with the data to get injection values from
    :param inj_func: The function that controls injection
    :return: List of injection data
    """
    injection_data = []
    for d in arr:
        injection_data.append(inj_func(d))
    injection_data = np.array(injection_data)
    return injection_data


def get_target_predictions(in_data, nn, inj_func=None):
    """
    Returns the target predictions obtained by using the neural network over in_data.
    :param in_data: The data that's used as in-data to the nn
    :param inj_func: The function used for the injection in nn
    :param nn: The neural network to use for predictions
    :return: A list of predictions for the target values
    """
    predictions = []
    if inj_func:    
        injection_data = inj_func(in_data)
    for i in range(len(in_data)):
        if inj_func:
            predictions.append(nn.predict(x=[np.array([in_data[i]]), np.array([injection_data[i]])]))
        else:
            predictions.append(nn.predict(x=[np.array([in_data[i]])]))

    return np.array(predictions)


def split_predictions(predictions, n):
    """
    Splits the data from predictions into n separate lists, so that we have n separate lists with all the values for each
    of the n variables.
    :param predictions: The list of predictions
    :param n: The number of variables
    :return: A list of lists with predicted values, one list for each variable
    """
    r_predictions = []
    for i in range(n):
        r_predictions.append(predictions[:, 0, i])
    return r_predictions


def split_values(targets, n):
    """
    Splits the data from targets into n separate lists,
    so that we have n separate lists with all the values for each of the n variables.
    :param targets: The list of targets
    :param n: The number of variables
    :return: A list of lists with actual values (targets), one list for each variable
    """
    actual_values = []
    for i in range(n):
        actual_values.append(targets[:, i])
    return actual_values

