import os
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from utils.load_data import concatenate_data
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import History, TensorBoard, ModelCheckpoint, Callback


class NBatchLogger(Callback):
    def __init__(self, display):
        super(Callback, self).__init__()
        self.seen = 0
        self.display = display

    def on_batch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        self.seen += logs.get('size', 0)
        if self.seen % self.display == 0:
            # you can access loss, accuracy in self.params['metrics']
            print('\n{}/{} - loss ....\n'.format(self.seen, self.params['nb_sample']))


def train_model(model, epoch, data, loss='mse', optimizer='rmsprop', save_best_only=True, metrics=None, show=False):
    # evaluation matrix
    metrics = ['mse'] if metrics is None else metrics
    model_name = model.name
    model_path = os.path.join(os.path.abspath(os.curdir), "saved_models", model_name, get_time_stamp())
    if not os.path.exists(model_path):
        os.makedirs(os.path.join(model_path, "saved_checkpoints"))
    # save the original dataset
    x_train, x_test, y_train, y_test = data
    with open(os.path.join(model_path, "data.pickle"), 'wb') as f:
        pickle.dump((x_train, x_test, y_train, y_test), f)
    # keras build in model structure visualization
    plot_model(model, to_file=os.path.join(model_path, f"{model_name}.png"))
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    # add keras callbacks save history, tensorboard record and checkpoints
    history = History()
    tensorboard = TensorBoard(log_dir=os.path.join(model_path, "logs"), update_freq="epoch")
    checkpoints = ModelCheckpoint(os.path.join(model_path, "saved_checkpoints", f"weights-{epoch:02d}.hdf5"),
                                  monitor='val_loss', mode='auto', save_freq='epoch', save_best_only=save_best_only)

    model.fit(x_train, y_train, validation_data=(x_test, y_test), callbacks=[history, tensorboard, checkpoints, ],
              epochs=epoch)
    # plot the history
    history_plot = plot_history(history.history, show=show)
    history_plot.savefig(os.path.join(model_path, f"{model_name}_Training_Error.png"))
    history_plot.close()
    # select the last model
    selected_file = os.listdir(os.path.join(model_path, "saved_checkpoints"))[-1]
    selected_model = load_check_point(os.path.join(model_path, "saved_checkpoints", selected_file))
    # plot the predicted value with the actual value
    x_origin, y_origin = concatenate_data(x_train, y_train, x_test, y_test)
    test_size = len(y_test)
    predict_plot = predict_and_plot(selected_model, x_origin, y_origin, test_size=test_size, show=show)
    predict_plot.savefig(os.path.join(model_path, f"{model_name}_Predicted_and_Actual.png"))
    predict_plot.close()
    with open(os.path.join(model_path, "history.pickle"), 'wb') as f:
        pickle.dump(history.history, f)


# load the saved checkpoint
def load_check_point(path):
    model = load_model(path)
    return model


# time stamp day-hour-minuts-second when running this function
def get_time_stamp():
    return time.strftime('%d-%H-%M-%S', time.localtime(time.time()))


# plot the predicted and actual value
def predict_and_plot(model, x, y, test_size=7, show=False):
    pred = model.predict(x)
    plt.figure()
    plt.plot(pred[:, 0])
    plt.plot(y)
    plt.vlines(len(y) - test_size, min(min(y), min(pred)), max(max(y), max(pred)), colors="r", linestyles="dashed")
    plt.title('Predicted and Actual')
    plt.ylabel('SMB')
    plt.xlabel('Year')
    plt.legend(['Predicted', 'Actual'], loc='upper left')
    if show:
        plt.show()
    return plt


# plot the training and validation loss history
def plot_history(history, show=False):
    plt.figure()
    plt.plot(np.log(history['loss']))
    plt.plot(np.log(history['val_loss']))
    plt.title('Loss and val_loss')
    plt.ylabel('Error')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    if show:
        plt.show()
    return plt


def load_and_plot_history(path):
    with open(path, 'rb') as f:
        history = pickle.load(f)
    plot_history(history, show=True)
