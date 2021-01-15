import os
import time
import pickle
import numpy as np
import pandas as pd
import importlib
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from utils.load_data import concatenate_data
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import History, TensorBoard, ModelCheckpoint
from utils.load_data import get_central, load_data_by_cluster, train_test_split


def train_model(model, epoch, data, loss='mse', optimizer='rmsprop', save_best_only=True, metrics=None, show=False):
    # evaluation matrix
    metrics = ['mse'] if metrics is None else metrics
    model_name = model.name
    model_path = os.path.join(os.path.abspath(os.curdir), "saved_models", model_name, get_time_stamp())
    if not os.path.exists(model_path):
        os.makedirs(model_path)
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

    model.fit(x_train, y_train, validation_data=(x_test, y_test), callbacks=[history, tensorboard, checkpoints],
              epochs=epoch)
    # plot the history
    history_plot = plot_history(history.history, show=show)
    history_plot.savefig(os.path.join(model_path, f"{model_name}_loss.png"))
    history_plot.close()
    # select the last model
    selected_file = os.listdir(os.path.join(model_path, "saved_checkpoints"))[-1]
    selected_model = load_check_point(os.path.join(model_path, "saved_checkpoints", selected_file))
    # plot the predicted value with the actual value
    x_origin, y_origin = concatenate_data(x_train, y_train, x_test, y_test)
    test_size = len(y_test)
    predict_plot = predict_and_plot(selected_model, x_origin, y_origin, test_size=test_size, show=show)
    predict_plot.savefig(os.path.join(model_path, f"{model_name}_value.png"))
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
    pred = pred[:, 0]
    plt.figure()
    plt.plot(pred)
    plt.plot(y)
    diff = pred - y
    train_diff, test_diff = diff[:-test_size], diff[-test_size:]
    loss, train_loss, test_loss = np.sum(np.power(diff, 2)) / len(diff), np.sum(np.power(train_diff, 2)) / len(
        train_diff), np.sum(np.power(test_diff, 2)) / len(test_diff)
    var, train_var, test_var = np.var(diff), np.var(train_diff), np.var(test_diff)
    std, train_std, test_std = np.std(diff), np.std(train_diff), np.std(test_diff)
    r2, train_r2, test_r2 = r2_score(y, pred), r2_score(y[:-test_size], pred[:-test_size]), \
                            r2_score(y[-test_size:], pred[-test_size:])
    df = pd.DataFrame({"name": [model.name], "loss": [loss], "var": [var], "std": [std], "r2_score": [r2],
                       "train_loss": [train_loss], "train_var": [train_var], "train_std": [train_std],
                       "train_r2": [train_r2],
                       "test_loss": [test_loss], "test_var": [test_var], "test_std": [test_std], "test_r2": [test_r2]
                       })
    if os.path.exists("loss_evaluate.csv"):
        eva = pd.read_csv("loss_evaluate.csv", index_col=0)
        eva = pd.concat([eva, df], sort=False)
        df = eva
    df.to_csv("loss_evaluate.csv")
    min_y, max_y = min(min(y), min(pred)), max(max(y), max(pred))
    plt.vlines(len(y) - test_size, min_y, max_y, colors="r", linestyles="dashed")
    plt.text(0, min(min(y), min(pred)), f'R2:{r2:.2f}, valR2:{test_r2:.2f}')
    plt.title(model.name)
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


def load_all_and_plot_all(saved_model_base_path, show=False):
    model_folders = os.listdir(saved_model_base_path)
    if "loss" in model_folders:
        model_folders.remove("loss")
    if not os.path.exists(os.path.join(saved_model_base_path, "loss")):
        os.makedirs(os.path.join(saved_model_base_path, "loss"))
    if "PredictedvsActual" in model_folders:
        model_folders.remove("PredictedvsActual")
    if not os.path.exists(os.path.join(saved_model_base_path, "PredictedvsActual")):
        os.makedirs(os.path.join(saved_model_base_path, "PredictedvsActual"))
    for model_name in model_folders:
        for model_index, running_time in enumerate(os.listdir(os.path.join(saved_model_base_path, model_name)), 1):
            base_path = os.path.join(saved_model_base_path, model_name, running_time)
            with open(os.path.join(base_path, "data.pickle"), 'rb') as f:
                (x_train, x_test, y_train, y_test) = pickle.load(f)
            x, y = concatenate_data(x_train, y_train, x_test, y_test)
            test_size = len(y_test)
            models_list = os.listdir(os.path.join(base_path, "saved_checkpoints"))
            if len(models_list) > 0:
                model = load_check_point(os.path.join(base_path, "saved_checkpoints",
                                                      models_list[-1]))
                pred_and_actual_plot = predict_and_plot(model, x, y, test_size=test_size, show=show)
                pred_and_actual_plot.savefig(os.path.join(saved_model_base_path, "PredictedvsActual",
                                                          f"{model_name}_value.png"))
                pred_and_actual_plot.close()
            if os.path.exists(os.path.join(base_path, "history.pickle")):
                with open(os.path.join(base_path, "history.pickle"), 'rb') as f:
                    history = pickle.load(f)
                history_plot = plot_history(history, show=show)
                history_plot.savefig(os.path.join(saved_model_base_path, "loss",
                                                  f"{model_name}_loss.png"))
                history_plot.close()
    print("Finished ploting")


def run_training(glaciers, category, model_names, glacier_df, centroid_map, epoch=2000, loss='mse', optimizer='rmsprop',
                 save_best_only=True, metrics=None, new_precipitation_path=None, combine=False):
    if combine:
        all_data = None
    for glacier in glaciers:
        try:
            central = get_central(glacier, glacier_df)
        except Exception as e:
            if "Central of " in str(e):
                print(str(e))
                continue
        print(f"Start training for glacier: {glacier} Central: {central}")
        try:
            if category == "basic":
                x_all, y_all = load_data_by_cluster(glacier, central, centroid_map,
                                                    "../Training_data/IGRA Archieves/",
                                                    "../Training_data/DMI_data",
                                                    "../Training_data/smb_mass_change.csv",
                                                    new_precipitation_path=new_precipitation_path)
            elif category == "ocean":
                x_all, y_all = load_data_by_cluster(glacier, central, centroid_map,
                                                    "../Training_data/IGRA Archieves/",
                                                    "../Training_data/DMI_data",
                                                    "../Training_data/smb_mass_change.csv",
                                                    ocean_surface_path="../Training_data/OceanSurface_observed",
                                                    new_precipitation_path=new_precipitation_path)
            elif category == "oceanreanalysis":
                x_all, y_all = load_data_by_cluster(glacier, central, centroid_map, "../Training_data/IGRA Archieves/",
                                                    "../Training_data/DMI_data", "../Training_data/smb_mass_change.csv",
                                                    ocean_surface_path="../Training_data/Ocean_Temperature_5m_Reanalysis",
                                                    new_precipitation_path=new_precipitation_path)
            else:
                raise Exception("Model groups not found")
        except FileNotFoundError:
            continue
        data_size = len(y_all)
        if data_size < 16 or np.sum(np.power(y_all, 2)) < 0.1:
            continue
        else:
            for x in x_all:
                print(x.shape, end=" ")
            print(y_all.shape)
            test_size = int(data_size * 0.2)
            x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=test_size)
            for module_name in model_names:
                module = importlib.import_module(module_name)
                model = module.getModel(f"{module_name}_{glacier[:15]}", height_range=x_all[3].shape[1],
                                        ocean_dim=x_all[-1].shape[1])
                print("#" * 20, "Start to train model: ", f"{module_name}_{glacier[:10]}", "#" * 20)
                train_model(model, epoch=epoch, data=(x_train, x_test, y_train, y_test),
                            loss=loss, optimizer=optimizer, save_best_only=save_best_only, metrics=metrics)
    print("Finished Training")
