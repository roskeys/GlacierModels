import os
import re
import numpy as np
import pandas as pd

index_matcher = re.compile(".*?/(\d{5})/?")


def load_1d_array(df, year_range):
    df = df[df['year'].isin(year_range)]
    return df[[str(i) for i in range(1, 13)]].values


def load_2d_array(df, year_range):
    df = df[df['year'].isin(year_range)]
    data = []
    for year in df["year"].unique():
        data.append(df[df["year"] == year][[str(i) for i in range(1, 13)]].values)
    return np.array(data)


def load_smb_array(df, year_range):
    df = df[df['year'].isin(year_range)]
    return df["SMB"].values


def get_common_year_range(year_range, df):
    return year_range.intersection(set(df["year"]))


def load_data(*paths):
    dataFrames = []
    for path in paths:
        dataFrames.append(pd.read_csv(path))
    year_range = set(dataFrames[0]["year"])
    for df in dataFrames:
        year_range = get_common_year_range(year_range, df)
    data = []
    smb = None
    for df in dataFrames:
        if df.shape[1] == 15:
            data.append(np.expand_dims(load_2d_array(df, year_range), axis=-1).astype(np.float))
        elif df.shape[1] == 14:
            data.append(load_1d_array(df, year_range).astype(np.float))
        elif df.shape[1] == 3:
            smb = load_smb_array(df, year_range).astype(np.float)
    return data, smb


def train_test_split(x, y, test_size=7, random_seed=None):
    if random_seed:
        np.random.seed(random_seed)
        shuffle_index = np.arange(len(y))
        np.random.shuffle(shuffle_index)
        for i, x_n in enumerate(x):
            x[i] = x_n[shuffle_index]
        y = y[shuffle_index]
    x_train, x_test = [], []
    for x_n in x:
        x_train.append(x_n[:-test_size])
        x_test.append(x_n[-test_size:])
    y_train = y[:-test_size]
    y_test = y[-test_size:]
    return x_train, x_test, y_train, y_test


def match_centroid(glacier_index, centroid_assignment):
    centroid_map = {5: 'AASIAAT(EGEDESMINDE)', 1: 'DANMARKSHAVN', 2: 'ITTOQQORTOORMIIT',
                    4: 'MITTARFIK_NARSARSUAQ', 3: 'TASIILAQ(AMMASSALIK)'}  # 'PITUFFIK',
    line = centroid_assignment[centroid_assignment["NAME"].str.contains(glacier_index)]
    if len(line) > 1:
        line = line.head(1)
    elif len(line) == 0:
        return
    return centroid_map[line["Central"].values[0]]


def load_data_of_glacier(dmi8_path, igra_path, smb_path):
    x, y = load_data(*[
        smb_path,
        os.path.join(dmi8_path, "cloud.csv"),
        os.path.join(dmi8_path, "precipitation.csv"),
        os.path.join(dmi8_path, "wind.csv"),
        os.path.join(igra_path, "humidity.csv"),
        os.path.join(igra_path, "pressure.csv"),
        os.path.join(igra_path, "temperature.csv"),
    ])
    return x, y


def load_data_by_cluster(cluster_path, smb_path):
    x, y = load_data(*[
        smb_path,
        os.path.join(cluster_path, "cloud.csv"),
        os.path.join(cluster_path, "precipitation.csv"),
        os.path.join(cluster_path, "wind.csv"),
        os.path.join(cluster_path, "humidity.csv"),
        os.path.join(cluster_path, "pressure.csv"),
        os.path.join(cluster_path, "temperature.csv"),
    ])
    return x, y


def concatenate_data(x1, y1, x2, y2):
    x_concatenated = []
    if isinstance(x1, list) or isinstance(x1, tuple):
        for x_part1, x_part2 in zip(x1, x2):
            x_concatenated.append(np.concatenate([x_part1, x_part2], axis=0))
    else:
        x_concatenated = np.concatenate([x1, x2], axis=0)
    y_concatenated = np.concatenate([y1, y2])
    return x_concatenated, y_concatenated


if __name__ == "__main__":
    all_x, all_y = load_data(*[
        "../data/MITTARFIK NARSARSUAQ/smb.csv",
        "../data/MITTARFIK NARSARSUAQ/mean_cloud.csv",
        "../data/MITTARFIK NARSARSUAQ/mean_precipitation.csv",
        "../data/MITTARFIK NARSARSUAQ/mean_wind.csv",
        "../data/MITTARFIK NARSARSUAQ/mean_temperature.csv",
        "../data/MITTARFIK NARSARSUAQ/mean_humidity.csv",
        "../data/MITTARFIK NARSARSUAQ/mean_pressure.csv"])
    x_train, x_test, y_train, y_test = train_test_split(all_x, all_y)
