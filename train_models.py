import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.abspath("."), "models", "basic"))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import importlib
import pandas as pd
from utils import train_model
from utils.load_data import train_test_split, load_data_of_glacier, match_centroid, load_data_by_cluster

model_files = os.listdir("models/basic")
model_files.remove("__init__.py")
if "__pycache__" in model_files:
    model_files.remove("__pycache__")
model_names = [n[:-3] for n in model_files]

for cluster in os.listdir("data/cluster"):
    x_all, y_all = load_data_by_cluster(f"data/cluster/{cluster}", "data/smb.csv")
    if len(y_all) < 25:
        continue
    test_size = int(len(y_all) * 0.2)
    x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=test_size)
    for module_name in model_names:
        module = importlib.import_module(module_name)
        model = module.getModel(f"{module_name}_{cluster}")
        print("#" * 20, "Start to train model: ", f"{module_name}_{cluster}", "#" * 20)
        train_model(model, epoch=2000, data=(x_train, x_test, y_train, y_test),
                    loss='mse', optimizer='rmsprop', save_best_only=True, metrics=['mse'])

# centroid_assignment = pd.read_csv("data/DMI_8_assignment.csv")
#
# model_files = os.listdir("models/basic")
# model_files.remove("__init__.py")
#
# if "__pycache__" in model_files:
#     model_files.remove("__pycache__")
#
# model_names = [n[:-3] for n in model_files]
# for glacier in os.listdir("data/DMI8_data"):
#     centroid = match_centroid(glacier, centroid_assignment)
#     x_all, y_all = load_data_of_glacier(f"data/DMI8_data/{glacier}", f"data/IGRA Archieves/{centroid}", "data/smb.csv")
#     if len(y_all) < 25:
#         continue
#     test_size = int(len(y_all) * 0.2)
#     x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=test_size)
#     for module_name in model_names:
#         module = importlib.import_module(module_name)
#         centroid = re.sub("\W", "", centroid)
#         model = module.getModel(f"{module_name}_{glacier}_{centroid}")
#         print("#" * 20, "Start to train model: ", f"{module_name}_{glacier}_{centroid}", "#" * 20)
#         train_model(model, epoch=2000, data=(x_train, x_test, y_train, y_test),
#                     loss='mse', optimizer='rmsprop', save_best_only=True, metrics=['mse'])
