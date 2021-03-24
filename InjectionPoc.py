import tensorflow as tf
import datetime
from settings import *
import numpy as np
from settings import NN_HIDDEN_LAYERS, LOSS, OPTIMIZER, INJECTION_LAYERS
import math
from typing import Union
from tqdm import tqdm


def get_predictions(starting_window, time_step, nn, terminal_time, injection_func=None):
    """
    Returns predictions over the time interval from starting_point to terminal_time (obtained by backfeeding) from the starting-point.
    :param starting_window: The starting-window (where to start predicting from)
    :param time_step: The time-step between two predictions
    :param nn: The nn to use for predicting
    :param terminal_time: Stop prediction when this time is reached
    :param injection_func: The function to use to calculate the injection (based on the whole window for a step)
    :return: A list of predictions
    """
    predictions = []
    last_window = starting_window
    last_step = last_window[-1]
    while round(last_step[-1], 2) < round(terminal_time, 2):
        if injection_func:
            prediction = nn.predict(x=[np.array([last_window]), np.array([injection_func(last_step)])])[0]
        else:
            prediction = nn.predict(x=[np.array([last_window])])[0]

        prediction = np.hstack((prediction, np.array([1])))
        new_step = last_step + prediction * time_step
        last_window = np.vstack((last_window[1:], new_step))
        predictions.append(new_step)
        last_step = new_step
        
    predictions = np.array(predictions)
    return predictions


def gen_input_layer(shape: tuple):
    """
    Generates a input layer for keras.
    The input layer is a node taking a dataset of size shape.
    :param shape: The shape of the input layer
    :return: The generated input layer
    """
    return tf.keras.layers.Input(shape=shape)


def gen_nn(hidden_layers: tuple, injection_nodes: dict, window: int, series_length: int, activation: str):
    """
    Generates a neural network of dense layers based on the params.
    :param hidden_layers: Tuple of hidden layer-dimensions. Implicitly determining number of hidden layers.
    :param injection_nodes: Dict of nodes to inject. K=layer (starting from 0), V=(n_nodes, injection_length)
    :param window: The length of the sliding window
    :param series_length: Length of a series for a input-node (data for one time step)
    :param activation: Activation function to use in hidden layers
    :return: The generated neural network.
    """
    # Generate nn
    input_size = (window, series_length)
    input_layer = gen_input_layer(input_size)
    x = input_layer
    injection_layers = []
    x = tf.keras.layers.Flatten()(x)

    # Iterate trough the layers and add them
    for i, dim in enumerate(hidden_layers):
        x = tf.keras.layers.Dense(dim, activation)(x)
        # If injection for this layer
        if i in injection_nodes.keys():
            # Create injection layer
            injection_layer = gen_input_layer(injection_nodes[i]["shape"])
            flatten_injection_layer = tf.keras.layers.Flatten()(injection_layer)

            # Add proxy level
            # x = tf.keras.layers.Dense(injection_nodes[i][0])(x)

            # Add injection level
            x = tf.keras.layers.Concatenate()([x, flatten_injection_layer])
            injection_layers.append(injection_layer)
    # Add output layer
    out = tf.keras.layers.Dense(series_length - 1)(x)
    model = tf.keras.models.Model(
        inputs=[input_layer] + injection_layers, outputs=out)
    return model


if __name__ == '__main__':
    nn = gen_nn(NN_HIDDEN_LAYERS, INJECTION_LAYERS, 2, 3)
    nn.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=None)
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1)
    x1 = np.array([[[1, 2, 3], [2, 2, 3]], [[1, 2, 4],
                                            [1, 2, 3]], [[1, 2, 3], [1, 2, 3]]])
    x2 = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    nn.fit(x=[x1, x2], y=np.array([[1, 0], [1, 0], [1, 0]]),
           epochs=3, callbacks=[tensorboard_callback])
