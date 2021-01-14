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
glaciers = glacier_df[glacier_df["Distance"] < 120]

final_df = pd.DataFrame()
for i in range(1, 6):
    central_i = glacier_df[glacier_df["Central"] == i].tail(3)
    final_df = pd.concat([final_df, central_i], ignore_index=True)

glaciers_set = final_df["NAME"].unique()
glaciers = [
    'JAKOBSHAVN_ISBRAE', 'DENDRITGLETSCHER', 'EQALORUTSIT_KILLIIT_SERMIAT', 'FENRISGLETSCHER', "SERMILIK",
    "SIORALIK-ARSUK-QIPISAQQU",
    "SORANERBRAEEN-EINAR_MIKKELSEN-HEINKEL-TVEGEGLETSCHER-PASTERZE", "STORSTROMMEN", "SW_NONAME1", "UKAASORSUAQ",
    "AB_DRACHMANN_GLETSCHER_L_BISTRUP_BRAE", "ADMIRALTY_TREFORK_KRUSBR_BORGJKEL_PONY", "KIATTUUT-QOOQQUP",
    "NAAJAT_SERMIAT", "SERMILIGAARSSUK_BRAE", "SYDBR", "HELHEIMGLETSCHER", "APUSEERAJIK", 'QAJUUTTAP_SERMIA',
    'KNUD-RASMUSSEN', "MIDGARDGLETSCHER", 'BREDEGLETSJER', "GEIKIE2", "NIGERTULUUP_KATTILERTARPIA",
]
for g in glaciers_set:
    glaciers.append(g)

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
