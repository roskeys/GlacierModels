import numpy as np
import pandas as pd


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
            smb = load_smb_array(df, year_range.astype(np.float))
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
