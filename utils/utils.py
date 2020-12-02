import os
import time
import pickle
import matplotlib.pyplot as plt
from utils.load_data import load_data, train_test_split
from tensorflow.keras.models import save_model, load_model
from tensorflow.keras.callbacks import History, TensorBoard, ModelCheckpoint
from tensorflow.keras.utils import plot_model


def train_model(model, epoch, loss='mse', optimizer='rmsprop', data_loader=load_data, save_best_only=True,
                train_test_spliter=train_test_split, test_size=7, random_state=42, matrics=None, plot=False,
                shuffle=False):
    matrics = ['mse'] if matrics is None else matrics
    model_name = model.name
    model_path = os.path.join(os.path.abspath(os.curdir), "models", model_name, get_time_stamp())
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        os.makedirs(os.path.join(model_path, "checkpoints"))
    plot_model(model, to_file=os.path.join(model_path, f"{model_name}.png"))
    model.compile(loss=loss, optimizer=optimizer, metrics=matrics)
    x, y = data_loader("data")
    x_train, y_train, x_test, y_test = train_test_spliter(x, y, test_size=test_size, random_state=random_state,
                                                          shuffle=shuffle)
    history = History()
    tensorboard = TensorBoard(log_dir=os.path.join(model_path, "logs"), update_freq="epoch")
    checkpoints = ModelCheckpoint(os.path.join(model_path, "checkpoints", "weights-{epoch:02d}.hdf5"),
                                  monitor='val_loss', mode='auto', save_freq='epoch', save_best_only=save_best_only)
    model.fit(x_train, y_train,
              validation_data=(x_test, y_test),
              callbacks=[history, tensorboard, checkpoints],
              epochs=epoch)
    save_model(model, os.path.join(model_path, "model.h5"))
    if plot:
        plot_history(history.history)
    plot_history(history.history, model_path)

    selected_file = os.listdir(os.path.join(model_path, "checkpoints"))[-1]
    selected_model = load_check_point(os.path.join(model_path, "checkpoints", selected_file))

    if plot:
        predict_and_plot(selected_model, x, y)
    predict_and_plot(selected_model, x, y, path=model_path)

    with open(os.path.join(model_path, "history.pickle"), 'wb') as f:
        pickle.dump(history.history, f)


def load_check_point(path):
    model = load_model(path)
    # print(model.summary())
    return model


def get_time_stamp():
    return time.strftime('%d-%H-%M-%S', time.localtime(time.time()))


def load_and_plot_history(path):
    with open(path, 'rb') as f:
        history = pickle.load(f)
    plot_history(history)


def predict_and_plot(model, x, y, path=None):
    pred = model.predict(x)
    plt.figure()
    plt.plot(pred[:, 0])
    plt.plot(y)
    plt.title('Predicted and Actual')
    plt.ylabel('SMB')
    plt.xlabel('Year')
    plt.legend(['Predicted', 'Actual'], loc='upper left')
    if path:
        plt.savefig(os.path.join(path, "Predicted_and_Actual.png"))
    else:
        plt.show()


def plot_history(history, path=None):
    plt.figure()
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


def pred_and_compare_side_by_side(path, x, y):
    model = load_model(path)
    predicted = model.predict(x)
    for pred, actual in zip(predicted, y):
        print(predicted, actual)
