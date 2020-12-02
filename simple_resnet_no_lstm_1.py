import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU, LSTM, MaxPooling2D, Conv2D, Flatten, concatenate
from tensorflow.keras.activations import relu
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

    cnn_1 = ResidualBlock(x, filters=8, kernel_size=3, strides=(1, 1), padding='same', shortcut=True)
    cnn_2 = ResidualBlock(cnn_1, filters=8, kernel_size=3, strides=(1, 1), padding='same', shortcut=True)
    cnn_3 = ResidualBlock(cnn_2, filters=8, kernel_size=3, strides=(1, 1), padding='same', shortcut=True)

    pool = MaxPooling2D(pool_size=(2, 2))(cnn_3)
    flattened = Flatten()(pool)
    lstm = Dense(64)(flattened)
    fc = LeakyReLU()(Dense(32)(lstm))
    pred = Dense(1)(fc)
    m = Model(inputs=[input_x1, input_x2, input_x3, input_x4], outputs=pred, name=name)
    return m


if __name__ == '__main__':
    path_name = os.path.basename(sys.argv[0])[:-3]
    from utils import train_model

    model = getModel(path_name)
    train_model(model, epoch=500, loss='mse', optimizer='rmsprop', test_size=7, random_state=42, shuffle=False, matrics=['mse'])
