import os
import numpy as np
import pandas as pd


def _load_1d_data(dataframe):
    feature_name = dataframe.columns[-1]
    year_range = dataframe["year"].unique()
    year_range.sort()
    for year in year_range[:-1]:
        data = []
        df_current_year = dataframe[dataframe["year"] == year]
        for i in range(5, 13):
            data.append(df_current_year.loc[df_current_year["month"] == i][feature_name].values)
        next_year = year + 1
        df_next_year = dataframe[dataframe["year"] == next_year]
        for i in range(1, 5):
            data.append(df_next_year.loc[df_next_year["month"] == i][feature_name].values)
        row = np.concatenate(data)
        assert row.shape == (12,)
        yield row
    return None


def _load_2d_data(dataframe):
    height = dataframe.shape[0]
    year_range = [int(year) for year in dataframe.columns.str.split("_").str[0].unique()[2:]]
    year_range.sort()
    for year in year_range[:-2]:
        data = []
        df_current_year = dataframe.filter(regex=f"^{year}", axis=1)
        for i in range(5, 13):
            record = df_current_year.filter(regex=f"_{i}$", axis=1)
            data.append(record.values)
        next_year = year + 1
        df_next_year = dataframe.filter(regex=f"^{next_year}", axis=1)
        for i in range(1, 5):
            record = df_next_year.filter(regex=f"_{i}$", axis=1)
            data.append(record.values)
        matrix = np.concatenate(data, axis=1)
        if matrix.shape != (height, 12):
            print(matrix.shape, year)
        yield np.expand_dims(matrix, 0)
    return None


def _get_1d_array(dataframe):
    return np.array(list(_load_1d_data(dataframe)))


def _get_2d_array(dataframe):
    return np.concatenate(list(_load_2d_data(dataframe)), axis=0)


def load_data(path):
    smb = pd.read_csv(os.path.join(path, "QAJUUTTAP_SERMIA_dm.csv"))["SMB"].values
    cloud = _get_1d_array(pd.read_csv(os.path.join(path, "mean_cloud.csv")))
    pressure = _get_2d_array(pd.read_csv(os.path.join(path, "Pressure_fill.csv")))
    wind = _get_1d_array(pd.read_csv(os.path.join(path, "mean_wind.csv")))
    humidity = _get_2d_array(pd.read_csv(os.path.join(path, "CalHum_std_fill.csv.csv")))
    temperature = _get_2d_array(pd.read_csv(os.path.join(path, "Temp_fill.csv")))
    precipitation = _get_1d_array(pd.read_csv(os.path.join(path, "monthly_total_precipitation.csv")))
    x = [np.concatenate([cloud, wind, precipitation], axis=1),
         np.expand_dims(humidity[:-9], -1),
         np.expand_dims(pressure[:-9], -1),
         np.expand_dims(temperature[:-9], -1)]
    return x, smb


def train_test_split(x, y, test_size=7, random_state=None):
    if random_state:
        np.random.seed(random_state)
    shuffle = np.arange(len(y))
    np.random.shuffle(shuffle)
    y = y[shuffle]
    x1 = x[0][shuffle]
    x2 = x[1][shuffle]
    x3 = x[2][shuffle]
    x4 = x[3][shuffle]
    return x1[test_size:], x2[test_size:], x3[test_size:], x4[test_size:], y[test_size:], \
           x1[:test_size], x2[:test_size], x3[:test_size], x4[:test_size], y[:test_size]


def data_generator(x1, x2, x3, x4, y):
    for i in range(len(y)):
        yield [x1[i], x2[i], x3[i], x4[i]], y[i]
    return None


if __name__ == '__main__':
    load_data("../data")
