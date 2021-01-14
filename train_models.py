import os
import sys

model_groups = ["basic", "ocean", "oceanreanalysis"]

for path in model_groups:
    sys.path.insert(0, os.path.join(os.path.abspath("."), "models", path))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
from utils.utils import load_all_and_plot_all, run_training

if os.name == "nt":
    epoch = 2
    if not os.path.exists("saved_models"):
        train = True
        plot = False
    else:
        train = False
        plot = True
else:
    epoch = 2000
    train = True
    plot = True

centroid_map = {5: 'AASIAAT(EGEDESMINDE)', 1: 'DANMARKSHAVN', 2: 'ITTOQQORTOORMIIT', 4: 'MITTARFIK_NARSARSUAQ',
                3: 'TASIILAQ(AMMASSALIK)'}  # 'PITUFFIK',

glacier_df = pd.read_csv("../Training_data/Glaicer_select.csv")

glaciers = [
    # cluster 5
    'JAKOBSHAVN_ISBRAE',  # 230 km
    "ICE_CAPS_SW",  # 177 km
    "SERMEQ_KUJALLEQ"  #
    # cluster 4
    "QAJUUTTAP_SERMIA",  # 52 km
    "SERMILIGAARSSUK_BRAE",  # 141 km
    # cluter 3
    "BUSSEMAND",  # 61 km
    "ICE_CAPS_CE",  # 173 km
    "HELHEIMGLETSCHER"  # 174
    # cluster 2
    "GEIKIE3",  # 88 km
    "DENDRITGLETSCHER",  # 193 km
    # cluster 1
    "AB_DRACHMANN_GLETSCHER_L_BISTRUP_BRAE",  # 149 km
    "STORSTROMMEN"  # 164
]

if train:
    for categories in model_groups:
        # get all the models
        model_files = os.listdir(f"models/{categories}")
        model_files.remove("__init__.py")
        if "__pycache__" in model_files:
            model_files.remove("__pycache__")
        model_names = [n[:-3] for n in model_files]
        run_training(glaciers, categories, model_names, glacier_df, centroid_map, epoch=epoch)
if plot:
    load_all_and_plot_all("saved_models", show=False)
