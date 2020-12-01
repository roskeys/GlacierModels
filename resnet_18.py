import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, ReLU, Conv2D, BatchNormalization, Input, add, Flatten, MaxPooling2D, \
    AveragePooling2D, concatenate


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
    print(x.shape)

    # conv1
    model = Conv2D(x, nb_filter=64, kernel_size=(7, 7), strides=(2, 2), padding="valid")
    model = BatchNormalization(axis=2)(model)
    model = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(model)

    # conv2
    model = ResidualBlock(model, nb_filter=64, kernel_size=(3, 3))
    model = ResidualBlock(model, nb_filter=64, kernel_size=(3, 3))

    # conv3
    model = ResidualBlock(model, nb_filter=128, kernel_size=(3, 3))
    model = ResidualBlock(model, nb_filter=128, kernel_size=(3, 3))

    # conv4
    model = ResidualBlock(model, nb_filter=256, kernel_size=(3, 3))
    model = ResidualBlock(model, nb_filter=256, kernel_size=(3, 3))

    # conv5
    model = ResidualBlock(model, nb_filter=512, kernel_size=(3, 3))
    model = ResidualBlock(model, nb_filter=512, kernel_size=(3, 3))

    # average
    model = AveragePooling2D(pool_size=(7, 7))(model)
    model = Flatten()(model)
    model = Dense(1)(model)

    m = Model(inputs=[input_x1, input_x2, input_x3, input_x4], outputs=pred, name=name)
    return m


# Defines the Residual Block, revised
def ResidualBlock(model, nb_filter, kernel_size, strides=(1, 1), padding='same', name=None):
    bn_name = (name + "_bn") if name != None else None
    conv_name = (name + "_conv") if name != None else None

    # BN->ReLU->Conv->BN->ReLU->Conv
    block = BatchNormalization(axis=2, name=bn_name)(model)
    block = ReLU()(block)
    block = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, name=conv_name)(block)

    block = BatchNormalization(axis=2, name=bn_name)(block)
    block = ReLU()(block)
    block = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, name=conv_name)(block)

    block = add([model, block])

    return block

if __name__ == '__main__':
    path_name = os.path.basename(sys.argv[0])[:-3]
    from utils import train_model

    model = getModel(path_name)
    train_model(model, epoch=100, loss='mse', optimizer='rmsprop', test_size=7, random_state=42, matrics=['mse'])