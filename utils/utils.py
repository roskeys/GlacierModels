import os
import time
import pickle
import matplotlib.pyplot as plt
from keras.callbacks import History, TensorBoard, ModelCheckpoint
from utils.load_data import load_data
from keras.utils.vis_utils import plot_model
from keras.models import save_model


def train_model(model, epoch, loss='mse', optimizer='rmsprop', validation_split=0.1, matrics=None):
    matrics = ['mse'] if matrics is None else matrics
    model_name = model.name
    model_path = os.path.join(os.path.abspath(os.curdir), "./models", model_name, get_time_stamp())
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    plot_model(model, to_file=os.path.join(model_path, f"{model_name}.png"))
    model.compile(loss=loss, optimizer=optimizer, metrics=matrics)
    x_1, (x_2, x_3, x_4), y = load_data("./data")
    history = History()
    tensorboard = TensorBoard(log_dir=os.path.join(model_path, "./logs"), update_freq="epoch")
    checkpoints = ModelCheckpoint(os.path.join(model_path, "checkpoint"), save_best_only=False, monitor="loss")
    model.fit((x_1, x_2, x_3, x_4), y, validation_split=validation_split, callbacks=[history, tensorboard, checkpoints],
              epochs=epoch)
    save_model(model, os.path.join(model_path, "model.h5"))
    plot_history(history.history)
    with open(os.path.join(model_path, "history.pickle"), 'wb') as f:
        pickle.dump(history.history, f)

def load_check_point(path):
    pass


def get_time_stamp():
    return time.strftime('%d-%H-%M-%S', time.localtime(time.time()))


def load_and_plot_history(path):
    with open(path, 'rb') as f:
        history = pickle.load(f)
    plot_history(history)


def plot_history(history):
    plt.plot(history['mse'])
    plt.plot(history['loss'])
    plt.title('Training Error')
    plt.ylabel('Error')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    plt.plot(history['val_mse'])
    plt.plot(history['val_loss'])
    plt.title('Test Error')
    plt.ylabel('Error')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
