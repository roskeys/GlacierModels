import os
import sys

model_groups = "basic"

sys.path.insert(0, os.path.join(os.path.abspath("."), "models", model_groups))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import importlib
import numpy as np
import pandas as pd
from utils.utils import train_model, load_all_and_plot_all
from utils.load_data import train_test_split, load_data_by_cluster, get_central

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
# get all the models
model_files = os.listdir(f"models/{model_groups}")
model_files.remove("__init__.py")
if "__pycache__" in model_files:
    model_files.remove("__pycache__")
model_names = [n[:-3] for n in model_files]
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
    for glacier in glaciers:
        central = get_central(glacier, glacier_df)
        print(f"Start training for glacier: {glacier} Central: {central}")
        if model_groups == "basic":
            x_all, y_all = load_data_by_cluster(glacier, central, centroid_map,
                                                "../Training_data/IGRA Archieves/",
                                                "../Training_data/DMI_data",
                                                "../Training_data/smb_mass_change.csv")
        elif model_groups == "ocean":
            x_all, y_all = load_data_by_cluster(glacier, central, centroid_map,
                                                "../Training_data/IGRA Archieves/",
                                                "../Training_data/DMI_data",
                                                "../Training_data/smb_mass_change.csv",
                                                ocean_surface_path="../Training_data/OceanSurface_observed")
        elif model_groups == "oceanreanalysis":
            x_all, y_all = load_data_by_cluster(glacier, central, centroid_map, "../Training_data/IGRA Archieves/",
                                                "../Training_data/DMI_data", "../Training_data/smb_mass_change.csv",
                                                ocean_surface_path="../Training_data/Ocean_Temperature_5m_Reanalysis")
        else:
            raise Exception("Model groups not found")
        data_size = len(y_all)
        if data_size < 20 or np.sum(np.power(y_all, 2)) < 0.1:
            continue
        else:
            for x in x_all:
                print(x.shape, end=" ")
            print(y_all.shape)
            test_size = int(data_size * 0.2)
            x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=test_size)
            for module_name in model_names:
                module = importlib.import_module(module_name)
                model = module.getModel(f"{module_name}_{glacier[:15]}")
                print("#" * 20, "Start to train model: ", f"{module_name}_{glacier[:10]}", "#" * 20)
                train_model(model, epoch=epoch, data=(x_train, x_test, y_train, y_test),
                            loss='mse', optimizer='rmsprop', save_best_only=True, metrics=['mse'])
    print("Finished Training")
if plot:
    load_all_and_plot_all("saved_models", show=False)
