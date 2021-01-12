import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.abspath("."), "models", "basic"))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import importlib
import pandas as pd
import numpy as np
from utils import train_model
from utils.load_data import train_test_split, load_data_by_cluster

centroid_map = {5: 'AASIAAT(EGEDESMINDE)', 1: 'DANMARKSHAVN', 2: 'ITTOQQORTOORMIIT', 4: 'MITTARFIK_NARSARSUAQ',
                3: 'TASIILAQ(AMMASSALIK)'}  # 'PITUFFIK',

# get all the models
model_files = os.listdir("models/basic")
model_files.remove("__init__.py")
if "__pycache__" in model_files:
    model_files.remove("__pycache__")
model_names = [n[:-3] for n in model_files]

for glacier in pd.read_csv("data/glacier_assignment.csv")["NAME"].unique():
    x_all, y_all = load_data_by_cluster("data/glacier_assignment.csv", glacier, centroid_map,
                                        "data/IGRA Archieves/", "data/newDMI8_data",
                                        "data/smb_mass_change.xlsx")
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
        train_model(model, epoch=2000, data=(x_train, x_test, y_train, y_test),
                    loss='mse', optimizer='rmsprop', save_best_only=True, metrics=['mse'])
