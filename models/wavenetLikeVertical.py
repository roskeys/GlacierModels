import tensorflow as tf
from tensorflow.keras import Model, Input
from components.ResNet import ResidualBlock
from tensorflow.keras.activations import relu
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, LSTM, concatenate


def getModel(name):
    # a training example is one dimensional vector 36 is the size
    input_x1 = Input(shape=(12,), name="cloud")
    input_x2 = Input(shape=(12,), name="precipitation")
    input_x3 = Input(shape=(12,), name="wind")
    input_x1_1 = tf.expand_dims(tf.expand_dims(input_x1, 1), -1)
    input_x1_2 = tf.expand_dims(tf.expand_dims(input_x2, 1), -1)
    input_x1_3 = tf.expand_dims(tf.expand_dims(input_x3, 1), -1)

    # a training example is 6 values a month,
    input_x4 = Input(shape=(40, 12, 1), name="Humidity")
    input_x5 = Input(shape=(40, 12, 1), name="Pressure")
    input_x6 = Input(shape=(40, 12, 1), name="Temperature")
    x = concatenate([input_x1_1, input_x1_2, input_x1_3, input_x4, input_x5, input_x6], axis=1)

    # 18 layer residuale block
    for _ in range(18):
        x = ResidualBlock(x, filters=16, kernel_size=3, strides=(1, 1), padding='same', shortcut=True)

    # 5 layer conv2D with horizontal kernel
    for _ in range(5):
        x = Conv2D(16, kernel_size=(2, 1), padding='valid', strides=(2, 1), activation=relu)(x)

    x = tf.expand_dims(Flatten()(x), -1)
    x = LSTM(128, activation=relu)(x)
    x = Dense(64, activation=relu)(x)
    x = Dropout(0.5)(x)
    pred = Dense(1)(x)

    m = Model(inputs=[input_x1, input_x2, input_x3, input_x4, input_x5, input_x6], outputs=pred, name=name)
    return m


if __name__ == '__main__':
    import os
    import sys

    sys.path.append("..")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    path_name = os.path.basename(sys.argv[0])[:-3]
    from utils import train_model
    from load_data_model_group_1 import load_data, train_test_split

    x_all, y_all = load_data(*[
        "../data/MITTARFIK NARSARSUAQ/smb.csv",
        "../data/MITTARFIK NARSARSUAQ/mean_cloud.csv",
        "../data/MITTARFIK NARSARSUAQ/mean_precipitation.csv",
        "../data/MITTARFIK NARSARSUAQ/mean_wind.csv",
        "../data/MITTARFIK NARSARSUAQ/mean_temperature.csv",
        "../data/MITTARFIK NARSARSUAQ/mean_humidity.csv",
        "../data/MITTARFIK NARSARSUAQ/mean_pressure.csv"])
    x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=7)
    model = getModel(path_name)
    train_model(model, epoch=1, data=(x_train, x_test, y_train, y_test),
                loss='mse', optimizer='rmsprop', save_best_only=True, matrics=['mse'])
