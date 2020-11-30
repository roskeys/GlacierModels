import os
import sys
from keras import Model
from keras.layers import Dense, Dropout, LeakyReLU, LSTM, MaxPooling2D, Conv2D, Flatten, concatenate
from keras.activations import relu
from keras import backend as K
from keras.backend import expand_dims
from keras.utils.vis_utils import plot_model
from utils import get_time_stamp, load_data, train_test_split, plot_history
from keras.callbacks import TensorBoard, History, ModelCheckpoint


def getModel(name):
    # a training example is one dimensional vector 36 is the size
    input_x1 = K.placeholder(shape=(None, 36,), name="cloud_wind_precipitation")
    # a training example is 6 values a month,
    input_x2 = K.placeholder(shape=(None, 40, 12, 1), name="Humidity")
    input_x3 = K.placeholder(shape=(None, 40, 12, 1), name="Pressure")
    input_x4 = K.placeholder(shape=(None, 40, 12, 1), name="Temperature")
    # nn model
    nn_1 = Dense(36, activation=relu)(input_x1)
    nn_1 = Dropout(0.5)(nn_1)

    # cnn layer 1 branch 1
    cnn_1_1 = Conv2D(4, kernel_size=(3, 3), padding='same', activation=relu)(input_x2)
    # cnn layer 1 branch 2
    cnn_1_2 = Conv2D(4, kernel_size=(3, 3), padding='same', activation=relu)(input_x3)
    # cnn layer 1 branch 3
    cnn_1_3 = Conv2D(4, kernel_size=(3, 3), padding='same', activation=relu)(input_x4)

    # cnn concat branches
    cnn_concat = concatenate([cnn_1_1, cnn_1_2, cnn_1_3], axis=-1)

    # cnn layer 2
    cnn_2 = Conv2D(3, kernel_size=(3, 3), padding='same', activation=relu, name="cnn_combine")(cnn_concat)
    pool = MaxPooling2D(pool_size=(2, 2))(cnn_2)
    flattened = Flatten()(pool)

    # joint two models
    x = concatenate([nn_1, flattened])
    x = expand_dims(x)
    lstm = LSTM(32)(x)
    fc = LeakyReLU()(Dense(24)(lstm))
    pred = Dense(1)(fc)
    m = Model(inputs=[input_x1, input_x2, input_x3, input_x4], outputs=pred, name=name)
    return m


if __name__ == '__main__':
    path_name = os.path.basename(sys.argv[0])[:-3]
    # from utils import train_model
    # train_model(model, epoch=10,  loss='mse', optimizer='rmsprop', validation_split=0.1, matrics=['mse'])

    model_path = os.path.join(os.path.abspath(os.curdir), "./models", path_name, get_time_stamp())
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model = getModel(path_name)
    model.compile(loss="mse", optimizer="adam", metrics=["mse"])

    x_1, (x_2, x_3, x_4), y = load_data("data")
    x1_train, x2_train, x3_train, x4_train, y_train, x1_test, x2_test, x3_test, x4_test, y_test = train_test_split(
        (x_1, x_2, x_3, x_4), y, test_size=7, random_state=42)
    # history = History()

    checkpoints = ModelCheckpoint(os.path.join(model_path, "checkpoint"), save_best_only=False, monitor="loss")
    model.fit((x_1, x_2, x_3, x_4), y, validation_split=0.1, callbacks=[checkpoints], epochs=10)
    results = model.evaluate((x1_test, x2_test, x3_test, x4_test), y_test)
    # plot_history(history.history)
