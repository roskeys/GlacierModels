import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.activations import tanh
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, concatenate, LSTM


def getModel(name):
    # a training example is one dimensional vector 36 is the size
    input_x1 = Input(shape=(12,), name="cloud")
    input_x2 = Input(shape=(12,), name="precipitation")
    input_x3 = Input(shape=(12,), name="wind")

    # a training example is 6 values a month,
    input_x4 = Input(shape=(41, 12, 1), name="Humidity")
    input_x5 = Input(shape=(41, 12, 1), name="Pressure")
    input_x6 = Input(shape=(41, 12, 1), name="Temperature")
    input_x7 = Input(shape=(447, 12, 1), name="Ocean")

    # nn model
    input_x_1 = concatenate([input_x1, input_x2, input_x3], axis=1)
    nn_1 = Dense(72, activation=tanh)(input_x_1)
    nn_1 = Dropout(0.5)(nn_1)

    # cnn layer 1 branch 1
    cnn_1_1 = Conv2D(16, kernel_size=(41, 1), padding='valid', activation=tanh)(input_x4)
    # cnn layer 1 branch 2
    cnn_1_2 = Conv2D(16, kernel_size=(41, 1), padding='valid', activation=tanh)(input_x5)
    # cnn layer 1 branch 3
    cnn_1_3 = Conv2D(16, kernel_size=(41, 1), padding='valid', activation=tanh)(input_x6)
    # cnn layer 1 branch 4
    cnn_1_4 = Conv2D(16, kernel_size=(9, 1), padding='valid', activation=tanh)(input_x7)

    # cnn layer 1 branch 1
    cnn_1_1 = Conv2D(4, kernel_size=(1, 2), padding='same', activation=tanh)(cnn_1_1)
    # cnn layer 1 branch 2
    cnn_1_2 = Conv2D(4, kernel_size=(1, 2), padding='same', activation=tanh)(cnn_1_2)
    # cnn layer 1 branch 3
    cnn_1_3 = Conv2D(4, kernel_size=(1, 2), padding='same', activation=tanh)(cnn_1_3)
    # cnn layer 1 branch 4
    cnn_1_4 = Conv2D(4, kernel_size=(1, 2), padding='same', activation=tanh)(cnn_1_4)

    # cnn concat branches
    cnn_concat = concatenate([cnn_1_1, cnn_1_2, cnn_1_3, cnn_1_4], axis=1)
    # cnn layer 2
    cnn_2 = Conv2D(4, kernel_size=(2, 1), padding='same', activation=tanh, name="cnn_combine")(cnn_concat)
    flattened = Flatten()(cnn_2)

    # joint two models
    x = concatenate([nn_1, flattened])
    x = tf.expand_dims(x, -1)
    lstm = LSTM(64)(x)
    fc = Dense(32)(lstm)
    pred = Dense(1)(fc)
    m = Model(inputs=[input_x1, input_x2, input_x3, input_x4, input_x5, input_x6, input_x7], outputs=pred, name=name)
    return m
