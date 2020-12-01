import os
import time
import pickle
import matplotlib.pyplot as plt
from utils.load_data import load_data, train_test_split
from tensorflow.keras.models import save_model, load_model
from tensorflow.keras.callbacks import History, TensorBoard, ModelCheckpoint
from tensorflow.keras.utils import plot_model


def train_model(model, epoch, loss='mse', optimizer='rmsprop', test_size=7, random_state=42, matrics=None, plot=False):
    matrics = ['mse'] if matrics is None else matrics
    model_name = model.name
    model_path = os.path.join(os.path.abspath(os.curdir), "models", model_name, get_time_stamp())
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        os.makedirs(os.path.join(model_path, "checkpoint"))
    plot_model(model, to_file=os.path.join(model_path, f"{model_name}.png"))
    model.compile(loss=loss, optimizer=optimizer, metrics=matrics)
    (x_1, x_2, x_3, x_4), y = load_data("data")
    x1_train, x2_train, x3_train, x4_train, y_train, x1_test, x2_test, x3_test, x4_test, y_test = train_test_split(
        (x_1, x_2, x_3, x_4), y, test_size=test_size, random_state=random_state)
    history = History()
    tensorboard = TensorBoard(log_dir=os.path.join(model_path, "logs"), update_freq="epoch")
    checkpoints = ModelCheckpoint(os.path.join(model_path, "checkpoint", "weights-{epoch:02d}.hdf5"),
                                  monitor='val_loss', mode='auto', save_freq='epoch')
    model.fit((x1_train, x2_train, x3_train, x4_train), y_train,
              validation_data=((x1_test, x2_test, x3_test, x4_test), y_test),
              callbacks=[history, tensorboard, checkpoints],
              epochs=epoch)
    save_model(model, os.path.join(model_path, "model.h5"))
    if plot:
        plot_history(history.history)
    else:
        plot_history(history.history, model_path)
    with open(os.path.join(model_path, "history.pickle"), 'wb') as f:
        pickle.dump(history.history, f)


def load_check_point(path):
    model = load_model(path)
    print(model.summary())
    return model


def get_time_stamp():
    return time.strftime('%d-%H-%M-%S', time.localtime(time.time()))


def load_and_plot_history(path):
    with open(path, 'rb') as f:
        history = pickle.load(f)
    plot_history(history)


def plot_history(history, path=None):
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Loss and val_loss')
    plt.ylabel('Error')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    if path:
        plt.savefig(os.path.join(path, "Training_Error.png"))
    else:
        plt.show()
