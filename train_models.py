import os
import sys

sys.path.append("models")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

tf.random.set_seed(1024)
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
model_names = [n[:-3] for n in model_files]

for module_name in model_names:
    module = importlib.import_module(f"models.{module_name}")
    model = module.getModel(module_name)
    train_model(model, epoch=2000, data=(x_train, x_test, y_train, y_test, x_all, y_all),
                loss='mse', optimizer='rmsprop', save_best_only=True, matrics=['mse'])
