import os
import sys
from keras import Model
from keras.layers import Dense, Dropout, ReLU, LSTM, MaxPooling2D, Conv2D, Flatten, concatenate, BatchNormalization, Input, add, Flatten, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras import backend as K
from keras.backend import expand_dims

def getModel(name, rows_num, column_num, channel = 1, classes):
    inpt = Input(shape = (rows_num, column_num, channel))
    model = ZeroPadding2D((3, 3))(inpt)

    # conv1
    model = Conv2D(model, nb_filter = 64, kernel_size = (7, 7), strides = (2, 2), padding = "valid")
    model = BatchNormalization(axis = 2)(model)
    model = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = "same")(model)

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
    model = AveragePooling2D(pool_size = (7, 7))(model)
    model = Flatten()(model)
    model = Dense(1)(model)

    return model

# Defines the Residual Block, revised
def ResidualBlock(model, nb_filter, kernel_size, strides = (1, 1), padding = 'same', name = None):
    bn_name = (name + "_bn") if name != None else None
    conv_name = (name + "_conv") if name != None else None

    # BN->ReLU->Conv->BN->ReLU->Conv
    block = BatchNormalization(axis = 2, name = bn_name)(model)
    block = ReLU()(block)
    block = Conv2D(nb_filter, kernel_size, padding = padding, strides = strides, name = conv_name)(block)

    block = BatchNormalization(axis = 2, name = bn_name)(block)
    block = ReLU()(block)
    block = Conv2D(nb_filter, kernel_size, padding = padding, strides = strides, name = conv_name)(block)

    block = add([model, block])

    return block