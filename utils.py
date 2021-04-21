import numpy as np
import tensorflow as tf

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


def get_target_predictions(in_data, nn, inj_layers=None):
    """
    Returns the target predictions obtained by using the neural network over in_data.
    :param in_data: The data that's used as in-data to the nn
    :param inj_layers: Dict that describes the layers that takes injection
    :param nn: The neural network to use for predictions
    :return: A list of predictions for the target values
    """
    predictions = []
    injection_data = None
    if inj_layers:
        injection_data = create_injection_data(inj_layers, in_data)
    for i in range(len(in_data)):
        if inj_layers:
            pred_injection = np.array([[d] for d in injection_data[:, i, :]])
            x_data = get_in_data_for_nn([in_data[i]], pred_injection)
            predictions.append(nn.predict(x=x_data))
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


def create_injection_data(injection_layers, data):
    """
    Creates a numpy array of injection data,
    based on the injection_layers (which define the injection function to use for each layer) and the input-data.
    :param injection_layers: The injection-layers
    :param data: The input-data
    :return: Numpy array of injection data for each injection-layer
    """
    return_data = []
    for layer in injection_layers.keys():
        inj_func = injection_layers[layer]["function"]
        return_data.append(inj_func(data))
    return np.array(return_data)


def get_in_data_for_nn(first_data, injection_data):
    in_data = [first_data]
    for d in injection_data:
        in_data.append(np.array(d))
    return in_data


def get_optimizer(opt_type, lr):
        """
        Returns an optimizer. The optimizer is specified by the name (i.e. 'adam', 'sgd') and learning-rate specified.
        :param opt_type: The name of optimizer type
        :param lr: The learning rate of the optimizer
        :return: The optimizer
        """
        if opt_type == 'adam':
            return tf.keras.optimizers.Adam(lr=lr)
        elif opt_type == 'adagrad':
            return tf.keras.optimizers.Adagrad(lr=lr)
        elif opt_type == 'sgd':
            return tf.keras.optimizers.SGD(lr=lr)
        elif opt_type == 'rms':
            return tf.keras.optimizers.RMSprop(lr=lr)
        else:
            raise ValueError('Invalid optimizer type: {}'.format(opt_type))

