from __future__ import absolute_import, division, print_function, unicode_literals

from tensorflow.keras.layers import SimpleRNN, Dense, Bidirectional, LSTM, Dropout
from tensorflow.keras import Sequential


def build_rnn_model(input_shape):
    """
    Builds a RNN model, compiles it and returns the model.
    :param input_shape: The shape of the dataset.
    :return: The RNN model
    """

    model = Sequential([
        LSTM(256, input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        LSTM(128),
        Dropout(0.2),
        Dense(units=2, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
