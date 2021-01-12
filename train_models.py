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

train = True
plot = False
if train:
    centroid_map = {5: 'AASIAAT(EGEDESMINDE)', 1: 'DANMARKSHAVN', 2: 'ITTOQQORTOORMIIT', 4: 'MITTARFIK_NARSARSUAQ',
                    3: 'TASIILAQ(AMMASSALIK)'}  # 'PITUFFIK',
    # get all the models
    model_files = os.listdir("models/basic")
    model_files.remove("__init__.py")
    if "__pycache__" in model_files:
        model_files.remove("__pycache__")
    model_names = [n[:-3] for n in model_files]

    glacier_df = pd.read_csv("../Training_data/Glaicer_select.csv")
    for glacier in glacier_df["NAME"].unique():
        x_all, y_all = load_data_by_cluster("../Training_data/Glaicer_select.csv", glacier, centroid_map,
                                            "../Training_data/IGRA Archieves/", "../Training_data/DMI_data",
                                            "../Training_data/smb_mass_change.csv")
        if len(y_all) < 25:
            continue
        if np.sum(np.power(y_all, 2)) < 0.1:
            continue
        test_size = int(len(y_all) * 0.2)
        x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=test_size)
        for module_name in model_names:
            module = importlib.import_module(module_name)
            model = module.getModel(f"{module_name}_{glacier}")
            print("#" * 20, "Start to train model: ", f"{module_name}_{glacier[:10]}", "#" * 20)
            train_model(model, epoch=2, data=(x_train, x_test, y_train, y_test),
                        loss='mse', optimizer='rmsprop', save_best_only=True, metrics=['mse'])
    print("Finished Training")

if plot:
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
            if os.path.exists(os.path.join(base_path, "history.pickle")):
                with open(os.path.join(base_path, "history.pickle"), 'rb') as f:
                    history = pickle.load(f)
                history_plot = plot_history(history, show=show)
                history_plot.savefig(os.path.join(saved_model_base_path, "loss",
                                                  f"{model_name}_Training_and_Evaluation_Loss.png"))
                history_plot.close()

    print("Finished ploting")

"""
Traceback (most recent call last):
  File "train_models.py", line 40, in <module>
    train_model(model, epoch=2, data=(x_train, x_test, y_train, y_test),
  File "utils.py", line 46, in train_model
    predict_plot.savefig(os.path.join(model_path, f"{model_name}_Predicted_and_Actual.png"))
  File "site-packages\matplotlib\pyplot.py", line 859, in savefig
    res = fig.savefig(*args, **kwargs)
  File "site-packages\matplotlib\figure.py", line 2311, in savefig
    self.canvas.print_figure(fname, **kwargs)
  File "site-packages\matplotlib\backend_bases.py", line 2210, in print_figure
    result = print_method(
  File "site-packages\matplotlib\backend_bases.py", line 1639, in wrapper
    return func(*args, **kwargs)
  File "site-packages\matplotlib\backends\backend_agg.py", line 510, in print_png
    mpl.image.imsave(
  File "site-packages\matplotlib\image.py", line 1611, in imsave
    image.save(fname, **pil_kwargs)
  File "site-packages\PIL\Image.py", line 2161, in save
    fp = builtins.open(filename, "w+b")
FileNotFoundError: [Errno 2] No such file or directory: 'saved_models\\cnnHorizontalLSTM_ADMIRALTY_TREFORK_KRUSBR_BORGJKEL_PONY\\12-22-01-54\\cnnHorizontalLSTM_ADMIRALTY_TREFORK_KRUSBR_BORGJKEL_PONY_Predicted_and_Actual.png'
"""
