import tensorflow as tf
import datetime
import numpy as np
from settings import NN_HIDDEN_LAYERS, LOSS, OPTIMIZER, INJECTION_LAYERS

a = [[0, 1, 2], [1, 2, 3]]
a = np.array(a)
print(a.shape)


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

model.fit(x=[x_train, x_train2],
          y=y_train,
          epochs=5,
          validation_data=([x_test, x_test], y_test),
          callbacks=[tensorboard_callback])
"""