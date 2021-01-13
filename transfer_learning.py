import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pandas as pd
from utils.continue_train import continue_train
from utils.load_data import train_test_split, load_data_by_cluster, get_central

model_selected = ["baselineModel_1_SW_NONAME1/13-18-52-32/saved_checkpoints/weights-02.hdf5"]

if os.name == "nt":
    epoch = 2
else:
    epoch = 2000

centroid_map = {5: 'AASIAAT(EGEDESMINDE)', 1: 'DANMARKSHAVN', 2: 'ITTOQQORTOORMIIT', 4: 'MITTARFIK_NARSARSUAQ',
                3: 'TASIILAQ(AMMASSALIK)'}  # 'PITUFFIK',

glacier_df = pd.read_csv("../Training_data/Glaicer_select.csv")
glaciers = glacier_df["NAME"].unique()

for glacier in glaciers:
    for model_path in model_selected:
        model_groups = model_path.split("_")[0]
        central = get_central(glacier, glacier_df)
        print(f"Start training for glacier: {glacier} Central: {central}")
        if model_groups == "ocean":
            x_all, y_all = load_data_by_cluster(glacier, central, centroid_map,
                                                "../Training_data/IGRA Archieves/",
                                                "../Training_data/DMI_data",
                                                "../Training_data/smb_mass_change.csv",
                                                ocean_surface_path="../Training_data/OceanSurface_observed")
        elif model_groups == "reanalysis":
            x_all, y_all = load_data_by_cluster(glacier, central, centroid_map, "../Training_data/IGRA Archieves/",
                                                "../Training_data/DMI_data", "../Training_data/smb_mass_change.csv",
                                                ocean_surface_path="../Training_data/Ocean_Temperature_5m_Reanalysis")
        else:
            x_all, y_all = load_data_by_cluster(glacier, central, centroid_map,
                                                "../Training_data/IGRA Archieves/",
                                                "../Training_data/DMI_data",
                                                "../Training_data/smb_mass_change.csv")
        data_size = len(y_all)
        if data_size < 16 or np.sum(np.power(y_all, 2)) < 0.1:
            continue
        else:
            for x in x_all:
                print(x.shape, end=" ")
            print(y_all.shape)
            test_size = int(data_size * 0.2)
            x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=test_size)
            model_path = os.path.join("saved_models", model_path)
            continue_train(model_path, epoch=epoch, data=(x_train, x_test, y_train, y_test),
                           loss='mse', optimizer='rmsprop', save_best_only=True, metrics=['mse'])
print("Finished Training")
