
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU, LSTM, MaxPooling2D, Conv2D, Flatten, concatenate
from tensorflow.keras.activations import relu, linear, tanh

def baseline_model(name):
    # a training example is one dimensional vector 36 is the size
    input_x1 = Input(shape=(36,), name="cloud_wind_precipitation")

    # a training example is 6 values a month,
    input_x2 = Input(shape=(40, 12, 1), name="Humidity")
    input_x3 = Input(shape=(40, 12, 1), name="Pressure")
    input_x4 = Input(shape=(40, 12, 1), name="Temperature")

    print(input_x1.shape, input_x4.shape)
    nn_1 = Dense(36, activation=relu)(input_x1)
    nn_1 = Dropout(0.5)(nn_1)

    # ann layer 1 branch 1
    nn_2 = Flatten()(input_x2)
    nn_2 = Dense(36, activation=relu)(nn_2)
    nn_2 = Dropout(0.5)(nn_2)
    # ann layer 1 branch 2
    nn_3 = Flatten()(input_x3)
    nn_3 = Dense(36, activation=relu)(nn_3)
    nn_3 = Dropout(0.5)(nn_3)
    # ann layer 1 branch 3
    nn_4 = Flatten()(input_x4)
    nn_4 = Dense(36, activation=relu)(nn_4)
    nn_4 = Dropout(0.5)(nn_4)

    # ann concat branches
    nn_concat = concatenate([nn_1, nn_2, nn_3, nn_4])

    # joint two models
    # x = tf.expand_dims(nn_concat, -1)
    dense_1 = Dense(32, activation=tanh)(nn_concat)
    # fc = LeakyReLU()(Dense(24)(dense_1))
    pred = Dense(1)(dense_1)
    m = Model(inputs=[input_x1, input_x2, input_x3, input_x4], outputs=pred, name=name)
    return m

if __name__ == '__main__':
    path_name = os.path.basename(sys.argv[0])[:-3]
    from utils import train_model
    model = baseline_model(path_name)
    train_model(model, epoch=1000, loss='mse', optimizer='rmsprop', test_size=7, matrics=['mse'])
