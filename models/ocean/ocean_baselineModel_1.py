from tensorflow.keras import Model, Input
from tensorflow.keras.activations import tanh
from tensorflow.keras.layers import Dense, Dropout, Flatten, concatenate


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

    input_x_1 = concatenate([input_x1, input_x2, input_x3], axis=1)
    nn_1 = Dense(72, activation=tanh)(input_x_1)
    nn_1 = Dropout(0.5)(nn_1)

    # ann layer 1 branch 1
    nn_2 = Flatten()(input_x4)
    nn_2 = Dense(72, activation=tanh)(nn_2)
    nn_2 = Dropout(0.5)(nn_2)
    # ann layer 1 branch 2
    nn_3 = Flatten()(input_x5)
    nn_3 = Dense(72, activation=tanh)(nn_3)
    nn_3 = Dropout(0.5)(nn_3)
    # ann layer 1 branch 3
    nn_4 = Flatten()(input_x6)
    nn_4 = Dense(72, activation=tanh)(nn_4)
    nn_4 = Dropout(0.5)(nn_4)

    nn_5 = Flatten()(input_x7)
    nn_5 = Dense(72, activation=tanh)(nn_5)
    nn_5 = Dropout(0.5)(nn_5)

    # ann concat branches
    nn_concat = concatenate([nn_1, nn_2, nn_3, nn_4, nn_5])

    # joint two models
    dense_1 = Dense(64, activation=tanh)(nn_concat)
    pred = Dense(1)(dense_1)
    m = Model(inputs=[input_x1, input_x2, input_x3, input_x4, input_x5, input_x6, input_x7], outputs=pred, name=name)
    return m
