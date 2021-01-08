import os
import sys

sys.path.insert(0, os.path.join(os.path.abspath("."), "models"))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import importlib
from utils import train_model
from models.load_data_model_group_1 import load_data, train_test_split

x_all, y_all = load_data(*[
    "data/MITTARFIK NARSARSUAQ/smb.csv",
    "data/MITTARFIK NARSARSUAQ/mean_cloud.csv",
    "data/MITTARFIK NARSARSUAQ/mean_precipitation.csv",
    "data/MITTARFIK NARSARSUAQ/mean_wind.csv",
    "data/MITTARFIK NARSARSUAQ/mean_temperature.csv",
    "data/MITTARFIK NARSARSUAQ/mean_humidity.csv",
    "data/MITTARFIK NARSARSUAQ/mean_pressure.csv"])

x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=7)
model_files = os.listdir("models")
model_files.remove("components")
model_files.remove("__init__.py")
model_files.remove("load_data_model_group_1.py")
if "__pycache__" in model_files:
    model_files.remove("__pycache__")

if "saved_models" in model_files:
    model_files.remove("saved_models")

model_names = [n[:-3] for n in model_files]
print(model_names)
for module_name in model_names:
    module = importlib.import_module(module_name)
    model = module.getModel(module_name)
    print("#" * 20, "Start to train model: ", module_name, "#" * 20)
    train_model(model, epoch=1, data=(x_train, x_test, y_train, y_test),
                loss='mse', optimizer='rmsprop', save_best_only=True, matrics=['mse'])
