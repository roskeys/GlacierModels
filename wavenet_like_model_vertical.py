import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, LSTM, concatenate
from tensorflow.keras.activations import tanh, relu
from resnet_18 import ResidualBlock


def getModel(name):
    # a training example is one dimensional vector 36 is the size
    input_x1 = Input(shape=(36,), name="cloud_wind_precipitation")
    input_x1_1 = tf.expand_dims(tf.expand_dims(input_x1[:, :12], 1), -1)
    input_x1_2 = tf.expand_dims(tf.expand_dims(input_x1[:, 12:24], 1), -1)
    input_x1_3 = tf.expand_dims(tf.expand_dims(input_x1[:, 24:36], 1), -1)

    # a training example is 6 values a month,
    input_x2 = Input(shape=(40, 12, 1), name="Humidity")
    input_x3 = Input(shape=(40, 12, 1), name="Pressure")
    input_x4 = Input(shape=(40, 12, 1), name="Temperature")

    x = concatenate([input_x2, input_x3, input_x4, input_x1_1, input_x1_2, input_x1_3], axis=1)

    for _ in range(18):
        x = ResidualBlock(x, filters=16, kernel_size=3, strides=(1, 1), padding='same', shortcut=True)

    for _ in range(5):
        x = Conv2D(16, kernel_size=(2, 1), padding='valid', strides=(2, 1), activation=relu)(x)

    x = tf.expand_dims(Flatten()(x), -1)
    x = LSTM(128, activation=relu)(x)
    x = Dense(64, activation=relu)(x)
    x = Dropout(0.5)(x)
    pred = Dense(1)(x)

    m = Model(inputs=[input_x1, input_x2, input_x3, input_x4], outputs=pred, name=name)
    return m


if __name__ == '__main__':
    path_name = os.path.basename(sys.argv[0])[:-3]
    from utils import train_model

    model = getModel(path_name)
    train_model(model, epoch=200, loss='mse', optimizer='rmsprop', test_size=7, random_state=42, matrics=['mse'])
