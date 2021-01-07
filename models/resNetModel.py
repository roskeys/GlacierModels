import tensorflow as tf
from tensorflow.keras import Model, Input
from components.ResNet import ResidualBlock
from tensorflow.keras.layers import Dense, LeakyReLU, MaxPooling2D, Flatten, concatenate


@tf.function
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
    for _ in range(8):
        x = ResidualBlock(x, filters=8, kernel_size=3, strides=(1, 1), padding='same', shortcut=True)
    pool = MaxPooling2D(pool_size=(2, 2))(x)
    flattened = Flatten()(pool)
    fc = Dense(64)(flattened)
    fc = LeakyReLU()(Dense(32)(fc))
    pred = Dense(1)(fc)
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
    train_model(model, epoch=10, data=(x_train, x_test, y_train, y_test, x_all, y_all),
                loss='mse', optimizer='rmsprop', save_best_only=True, matrics=['mse'])
