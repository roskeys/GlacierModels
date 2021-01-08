import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pickle
from utils import load_check_point, plot_history, predict_and_plot

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
            (x, y) = pickle.load(f)
        model = load_check_point(
            os.path.join(base_path, "saved_checkpoints", os.listdir(os.path.join(base_path, "saved_checkpoints"))[-1]))
        pred_and_actual_plot = predict_and_plot(model, x, y, show=show)
        pred_and_actual_plot.savefig(
            os.path.join(saved_model_base_path, "PredictedvsActual", f"{model_name}_Predicted_and_Actual.png"))
        with open(os.path.join(base_path, "history.pickle"), 'rb') as f:
            history = pickle.load(f)
        history_plot = plot_history(history, show=show)
        history_plot.savefig(
            os.path.join(saved_model_base_path, "loss", f"{model_name}_Training_and_Evaluation_Loss.png"))
