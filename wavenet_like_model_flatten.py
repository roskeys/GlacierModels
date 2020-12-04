import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Conv2D, Conv1D, Flatten, Dropout, LSTM, concatenate
from tensorflow.keras.activations import relu, relu
from resnet_18 import ResidualBlock


def getModel(name):
    # a training example is one dimensional vector 36 is the size
    input_x1 = Input(shape=(36,), name="cloud_wind_precipitation")
    # a training example is 6 values a month,
    input_x2 = Input(shape=(40, 12, 1), name="Humidity")
    input_x3 = Input(shape=(40, 12, 1), name="Pressure")
    input_x4 = Input(shape=(40, 12, 1), name="Temperature")

    x = concatenate([
        concatenate([tf.gather(input_x1, [0, 12, 24], axis=1),
                     tf.squeeze(input_x2[:, :, i, :], axis=-1),
                     tf.squeeze(input_x3[:, :, i, :], axis=-1),
                     tf.squeeze(input_x4[:, :, i, :], axis=-1)],
                    axis=1) for i in range(12)], axis=1)
    x = tf.expand_dims(x, -1)
    for _ in range(6):
        x = Conv1D(16, kernel_size=2, padding='valid', strides=2, activation=relu)(x)
    x = Flatten()(x)
    x = tf.expand_dims(x, -1)
    x = LSTM(128)(x)
    x = Dense(64, activation=relu)(x)
    x = Dropout(0.5)(x)
    pred = Dense(1)(x)
    m = Model(inputs=[input_x1, input_x2, input_x3, input_x4], outputs=pred, name=name)
    return m


if __name__ == '__main__':
    path_name = os.path.basename(sys.argv[0])[:-3]
    from utils import train_model

    model = getModel(path_name)
    train_model(model, epoch=2000, loss='mse', optimizer='rmsprop', test_size=7, random_state=42, matrics=['mse'])
