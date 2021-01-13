import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.activations import tanh
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, LSTM, concatenate


def getModel(name):
    # a training example is one dimensional vector 36 is the size
    input_x1 = Input(shape=(12,), name="cloud")
    input_x2 = Input(shape=(12,), name="precipitation")
    input_x3 = Input(shape=(12,), name="wind")

    # a training example is 6 values a month,
    input_x4 = Input(shape=(41, 12, 1), name="Humidity")
    input_x5 = Input(shape=(41, 12, 1), name="Pressure")
    input_x6 = Input(shape=(41, 12, 1), name="Temperature")
    input_x7 = Input(shape=(9, 12, 1), name="Ocean")

    # flatten the dataset to 1D
    x = concatenate(
        [Flatten()(input_x1), Flatten()(input_x2), Flatten()(input_x3),
         Flatten()(input_x4), Flatten()(input_x5), Flatten()(input_x6), Flatten()(input_x7)])
    x = tf.expand_dims(x, -1)

    # 10 layer of conv 1D with kernel size 2
    for _ in range(10):
        x = Conv1D(16, kernel_size=2, padding='valid', strides=2, activation=tanh)(x)

    x = Flatten()(x)
    x = tf.expand_dims(x, -1)
    x = LSTM(64)(x)
    x = Dense(32, activation=tanh)(x)
    x = Dropout(0.5)(x)
    pred = Dense(1)(x)
    m = Model(inputs=[input_x1, input_x2, input_x3, input_x4, input_x5, input_x6, input_x7], outputs=pred, name=name)
    return m
