import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.activations import relu
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU, LSTM, Conv2D, Flatten, concatenate


def getModel(name):
    # a training example is one dimensional vector 36 is the size
    input_x1 = Input(shape=(12,), name="cloud")
    input_x2 = Input(shape=(12,), name="precipitation")
    input_x3 = Input(shape=(12,), name="wind")

    # a training example is 6 values a month,
    input_x4 = Input(shape=(41, 12, 1), name="Humidity")
    input_x5 = Input(shape=(41, 12, 1), name="Pressure")
    input_x6 = Input(shape=(41, 12, 1), name="Temperature")

    # nn model
    input_x_1 = concatenate([input_x1, input_x2, input_x3], axis=1)
    nn_1 = Dense(36, activation=relu)(input_x_1)
    nn_1 = Dropout(0.5)(nn_1)

    # cnn layer 1 branch 1
    cnn_1_1 = Conv2D(4, kernel_size=(1, 12), padding='valid', activation=relu)(input_x4)
    # cnn layer 1 branch 2
    cnn_1_2 = Conv2D(4, kernel_size=(1, 12), padding='valid', activation=relu)(input_x5)
    # cnn layer 1 branch 3
    cnn_1_3 = Conv2D(4, kernel_size=(1, 12), padding='valid', activation=relu)(input_x6)
    # cnn concat branches
    cnn_concat = concatenate([cnn_1_1, cnn_1_2, cnn_1_3], axis=3)
    # cnn layer 2
    cnn_2 = Conv2D(3, kernel_size=(2, 1), padding='same', activation=relu, name="cnn_combine")(cnn_concat)
    flattened = Flatten()(cnn_2)

    # joint two models
    x = concatenate([nn_1, flattened])
    x = tf.expand_dims(x, -1)
    lstm = LSTM(32)(x)
    fc = LeakyReLU()(Dense(24)(lstm))
    pred = Dense(1)(fc)
    m = Model(inputs=[input_x1, input_x2, input_x3, input_x4, input_x5, input_x6], outputs=pred, name=name)
    return m
