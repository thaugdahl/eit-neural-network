import tensorflow as tf
import datetime
from settings import *
import numpy as np
import math
from typing import Union

a = [[0, 1, 2], [1, 2, 3]]
a = np.array(a)
print(a.shape)

"""
Example Sliding Window
Sliding window size = 2

Single timestep samples from cos(x), x=t*pi/4
"""
samples = [
    [
        # [x0, y0, t0,] , [x1, y1, t1]
        [0, 1, 0], [math.pi / 4, 0.5, 1]
    ],
    [
        # [x1, y1, t1,] , [x2, y2, t2]
        [math.pi / 4, 0.5, 1], [math.pi / 2, 0, 2]
    ],
    [
        # [x2, y2, t2,] , [x3, y3, t3]
        [math.pi / 2, 0, 2], [(3*math.pi) / 4, -0.5, 3]
    ],
    [
        # [x3, y3, t3,] , [x4, y4, t4]
        [(3*math.pi) / 4, -0.5, 3], [math.pi, -1, 4]
    ]
]


def predict(
        nn: tf.keras.models.Model,
        window: Union[np.ndarray, list],
        steps: int,
        delta_t: float,
        values_per_step: int = 3,
        steps_per_window: int = 2
):
    """
    :return: History of single timestep samples including the window used in this function
    """
    global samples
    history = np.array(window, ndmin=1)
    window = np.array(window)
    history = np.append(history, [samples[1][1]], axis=0)

    print(history[-SLIDING_WINDOW_LENGTH:])

    # Use of nn.predict() to get the derivatives
    for i in range(steps):
        # beregene nye verdier for injected layers
        # og feede de til riktige inputs
        derivatives = nn.predict(x=[history[-SLIDING_WINDOW_LENGTH:]])
        # derivatives = nn.predict(x=[history[-SLIDING_WINDOW_LENGTH:], injected_array])
        # Sørge for å ta ut t herfra først.
        new_values = history[-1:][0:DATA_NUM_VARIABLES] + derivatives * delta_t
        new_t = history[-1][-1] + delta_t

        new_single_sample = np.array(new_values + new_t)
        np.insert(history, new_single_sample)

    return history

def gen_input_layer(series_length: int):
    """
    Generates a input layer for keras.
    The input layer is a node taking a 1-D dataset of length series_length
    :param series_length: The length of the dataset for one node
    :return: The generated input layer
    """
    shape = (series_length, )
    return tf.keras.layers.Input(shape=shape)


def gen_nn(hidden_layers: tuple, injection_nodes: dict, series_length: int):
    """
    Generates a neural network of dense layers based on the params.
    :param hidden_layers: Tuple of hidden layer-dimensions. Implicitly determining number of hidden layers.
    :param injection_nodes: Dict of nodes to inject. K=layer (starting from 0), V=(n_nodes, injection_length)
    :param series_length: Length of a series for a input-node
    :return: The generated neural network.
    """
    # Generate nn
    input_layer = gen_input_layer(series_length)
    x = input_layer
    injection_layers = []
    for i, dim in enumerate(hidden_layers):
        x = tf.keras.layers.Dense(dim)(x)
        if i in injection_nodes.keys():
            # Create injection layer
            injection_node, injection_length = injection_nodes[i]
            injection_layer = gen_input_layer(injection_length)
            # Add proxy level
            x = tf.keras.layers.Dense(injection_length)(x)
            # Add injection level
            x = tf.keras.layers.Add()([x, injection_layer])
            injection_layers.append(injection_layer)
    out = tf.keras.layers.Dense(series_length - 1)(x)
    model = tf.keras.models.Model(inputs=[input_layer] + injection_layers, outputs=out)
    return model


if __name__ == '__main__':
    nn = gen_nn((64, 128, 64), {2: (1, 3)}, 3)
    # nn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=None)
    #log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    #x1 = np.array([[1, 2, 3], [1, 2, 3]])
    #x2 = np.array([[1, 2, 3], [1, 2, 3]])
    #nn.fit(x=[x1, x2], y=np.array([[1,], [1, ]]), epochs=3, callbacks=[tensorboard_callback])
    predict(nn, samples[0], 3, 1, 3, 2)

