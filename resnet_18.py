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
    model = Conv2D(filters=8, kernel_size=(3, 3), padding="valid")(x)
    model = BatchNormalization(axis=3)(model)
    model = MaxPooling2D(pool_size=(3, 3), padding="same")(model)

    # conv2
    model = ResidualBlock(model, filters=8, kernel_size=(3, 3))
    model = ResidualBlock(model, filters=8, kernel_size=(3, 3))

    # conv3
    model = ResidualBlock(model, filters=16, kernel_size=(3, 3), shortcut = True)
    model = ResidualBlock(model, filters=16, kernel_size=(3, 3))

    # conv4
    model = ResidualBlock(model, filters=32, kernel_size=(3, 3) ,shortcut= True)
    model = ResidualBlock(model, filters=32, kernel_size=(3, 3))

    # conv5
    model = ResidualBlock(model, filters=64, kernel_size=(3, 3), shortcut = True)
    model = ResidualBlock(model, filters=64, kernel_size=(3, 3))

    # average
    model = AveragePooling2D(pool_size=(2, 2))(model)
    model = Flatten()(model)
    model = Dense(1)(model)

    m = Model(inputs=[input_x1, input_x2, input_x3, input_x4], outputs=model, name=name)
    return m


# Defines the Residual Block, revised
def ResidualBlock(model, filters, kernel_size, strides=(1, 1), padding='same', name=None, shortcut = False):
    bn_name = (name + "_bn") if name != None else None
    conv_name = (name + "_conv") if name != None else None

    # Conv->ReLU->BN
    block = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(model)
    block = BatchNormalization(axis=3, name=bn_name)(block)

    block = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(block)
    block = BatchNormalization(axis=3, name=bn_name)(block)

    if shortcut:
        shortcutBlock = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(model)
        shortcutBlock = BatchNormalization(axis=3, name=bn_name)(shortcutBlock)
        block = add([block, shortcutBlock])
    else:
        block = add([model, block])

    return block

if __name__ == '__main__':
    path_name = os.path.basename(sys.argv[0])[:-3]
    from utils import train_model

    model = getModel(path_name)
    train_model(model, epoch=100, loss='mse', optimizer='rmsprop', test_size=7, random_state=42, matrics=['mse'])