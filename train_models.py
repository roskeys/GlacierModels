import os
import sys

sys.path.insert(0, os.path.join(os.path.abspath("."), "models", "basic"))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pickle
import importlib
import numpy as np
import pandas as pd
from utils import load_check_point, plot_history, predict_and_plot, train_model
from utils.load_data import train_test_split, load_data_by_cluster, concatenate_data

centroid_map = {5: 'AASIAAT(EGEDESMINDE)', 1: 'DANMARKSHAVN', 2: 'ITTOQQORTOORMIIT', 4: 'MITTARFIK_NARSARSUAQ',
                3: 'TASIILAQ(AMMASSALIK)'}  # 'PITUFFIK',

# get all the models
model_files = os.listdir("models/basic")
model_files.remove("__init__.py")
if "__pycache__" in model_files:
    model_files.remove("__pycache__")
model_names = [n[:-3] for n in model_files]

glacier_df = pd.read_csv("data/smb_mass_change.csv")
glacier_df = glacier_df[glacier_df["METHOD"] == "SMB"]

for glacier in glacier_df["NAME"].unique():
    x_all, y_all = load_data_by_cluster("data/glacier_assignment.csv", glacier, centroid_map,
                                        "data/IGRA Archieves/", "data/newDMI8_data",
                                        "data/smb_mass_change.csv")
    if len(y_all) < 25:
        continue
    if np.sum(np.power(y_all, 2)) < 0.1:
        continue
    test_size = int(len(y_all) * 0.2)
    x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=test_size)
    for module_name in model_names:
        module = importlib.import_module(module_name)
        model = module.getModel(f"{module_name}_{glacier}")
        print("#" * 20, "Start to train model: ", f"{module_name}_{glacier}", "#" * 20)
        train_model(model, epoch=2, data=(x_train, x_test, y_train, y_test),
                    loss='mse', optimizer='rmsprop', save_best_only=True, metrics=['mse'])

saved_model_base_path = "saved_models"
show = False
models = []

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
                                                      f"{model_name}_Predicted_and_Actual.png"))
            pred_and_actual_plot.close()
            with open(os.path.join(base_path, "history.pickle"), 'rb') as f:
                history = pickle.load(f)
            history_plot = plot_history(history, show=show)
            history_plot.savefig(os.path.join(saved_model_base_path, "loss",
                                              f"{model_name}_Training_and_Evaluation_Loss.png"))
            history_plot.close()
