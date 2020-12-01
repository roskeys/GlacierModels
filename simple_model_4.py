import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU, LSTM, MaxPooling2D, Conv2D, Flatten, concatenate
from tensorflow.keras.activations import relu


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

    # nn model
    nn_1 = Dense(36, activation=relu)(input_x1)
    nn_1 = Dropout(0.5)(nn_1)

    # cnn layer 2
    cnn_2 = Conv2D(3, kernel_size=(3, 3), padding='same', activation=relu, name="cnn_combine")(x)
    pool = MaxPooling2D(pool_size=(2, 2))(cnn_2)
    flattened = Flatten()(pool)

    # joint two models
    x = concatenate([nn_1, flattened])
    x = tf.expand_dims(x, -1)
    lstm = LSTM(32)(x)
    fc = LeakyReLU()(Dense(24)(lstm))
    pred = Dense(1)(fc)
    m = Model(inputs=[input_x1, input_x2, input_x3, input_x4], outputs=pred, name=name)
    return m


if __name__ == '__main__':
    path_name = os.path.basename(sys.argv[0])[:-3]
    from utils import train_model

    model = getModel(path_name)
    train_model(model, epoch=200, loss='mse', optimizer='rmsprop', test_size=7, random_state=42, matrics=['mse'])
