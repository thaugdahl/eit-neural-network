import tensorflow as tf
import datetime
from settings import *
import numpy as np
from settings import NN_HIDDEN_LAYERS, LOSS, OPTIMIZER, INJECTION_LAYERS
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

def gen_input_layer(shape: tuple):
    """
    Generates a input layer for keras.
    The input layer is a node taking a dataset of size shape.
    :param shape: The shape of the input layer
    :return: The generated input layer
    """
    return tf.keras.layers.Input(shape=shape)


def gen_nn(hidden_layers: tuple, injection_nodes: dict, window: int, series_length: int):
    """
    Generates a neural network of dense layers based on the params.
    :param hidden_layers: Tuple of hidden layer-dimensions. Implicitly determining number of hidden layers.
    :param injection_nodes: Dict of nodes to inject. K=layer (starting from 0), V=(n_nodes, injection_length)
    :param window: The length of the sliding window
    :param series_length: Length of a series for a input-node (data for one time step)
    :return: The generated neural network.
    """
    # Generate nn
    input_size = (window, series_length)
    input_layer = gen_input_layer(input_size)
    x = input_layer
    injection_layers = []
    # Iterate trough the layers and add them
    for i, dim in enumerate(hidden_layers):
        x = tf.keras.layers.Dense(dim)(x)
        # If injection for this layer
        if i in injection_nodes.keys():
            # Create injection layer
            injection_layer = gen_input_layer(injection_nodes[i])
            # Add proxy level
            x = tf.keras.layers.Dense(injection_nodes[i][0])(x)
            # Add injection level
            x = tf.keras.layers.Add()([x, injection_layer])
            injection_layers.append(injection_layer)
    # Add output layer
    out = tf.keras.layers.Dense(series_length - 1)(x)
    model = tf.keras.models.Model(inputs=[input_layer] + injection_layers, outputs=out)
    return model


if __name__ == '__main__':
    nn = gen_nn(NN_HIDDEN_LAYERS, INJECTION_LAYERS, 2, 3)
    nn.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=None)
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    x1 = np.array([[[1, 2, 3], [2, 2, 3]], [[1, 2, 4], [1, 2, 3]]])
    x2 = np.array([[1, 2, 3], [1, 2, 3]])
    nn.fit(x=[x1, x2], y=np.array([[1, 0], [1, 0]]), epochs=3, callbacks=[tensorboard_callback])

"""
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train2 = x_train

input1 = tf.keras.layers.Input(shape=(28, 28))
x1 = tf.keras.layers.Dense(8, activation='relu')(input1)
dense2 = tf.keras.layers.Dense(28)(x1)
input2 = tf.keras.layers.Input(shape=(28, 28))
added = tf.keras.layers.Add()([dense2, input2])
out = tf.keras.layers.Dense(1)(added)
model = tf.keras.models.Model(inputs=[input1, input2], outputs=out)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    nn = gen_nn((64, 128, 64), {2: (1, 3)}, 3)
    # nn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=None)
    #log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    #x1 = np.array([[1, 2, 3], [1, 2, 3]])
    #x2 = np.array([[1, 2, 3], [1, 2, 3]])
    #nn.fit(x=[x1, x2], y=np.array([[1,], [1, ]]), epochs=3, callbacks=[tensorboard_callback])
    predict(nn, samples[0], 3, 1, 3, 2)
"""

