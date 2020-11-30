import os
import numpy as np
import pandas as pd


def generator_1d(dataframe):
    feature_name = dataframe.columns[-1]
    for year in dataframe["year"].unique():
        df = dataframe[dataframe["year"] == year]
        if df.shape[0] == 12:
            data = df[feature_name].values
            yield data
        elif df.shape[0] < 12:
            d = {}
            for row in df.iterrows():
                d[row[1][2]] = row[1][-1]
            data = []
            for i in range(1, 13):
                if i in d.keys():
                    data.append(d[i])
                else:
                    data.append(0)
            yield np.array(data)
        else:
            assert False
    return None


def generator_2d(dataframe):
    height = dataframe.shape[0]
    year_range = dataframe.columns.str.split("_").str[0].unique()[2:]
    for year in year_range:
        df = dataframe.filter(regex="^" + year, axis=1)
        d = {}
        for column in df.columns:
            d[int(column.split("_")[-1])] = df[column].values.T
        data = []
        for i in range(1, 13):
            if i in d.keys():
                data.append(d[i].reshape(height, 1))
            else:
                data.append(np.zeros((height, 1)))
        yield np.expand_dims(np.concatenate(data, axis=1), 0)
    return None


def get_1d_array(dataframe):
    return np.array(list(generator_1d(dataframe)))


def get_2d_array(dataframe):
    return np.concatenate(list(generator_2d(dataframe)), axis=0)


def load_data(path):
    smb = pd.read_csv(os.path.join(path, "QAJUUTTAP_SERMIA_dm.csv"))["SMB"].values
    cloud = get_1d_array(pd.read_csv(os.path.join(path, "mean_cloud.csv")))
    pressure = get_2d_array(pd.read_csv(os.path.join(path, "Pressure_fill.csv")))
    wind = get_1d_array(pd.read_csv(os.path.join(path, "mean_wind.csv")))
    humidity = get_2d_array(pd.read_csv(os.path.join(path, "CalHum_std_fill.csv.csv")))
    temperature = get_2d_array(pd.read_csv(os.path.join(path, "Temp_fill.csv")))
    precipitation = get_1d_array(pd.read_csv(os.path.join(path, "monthly_total_precipitation.csv")))
    y = smb
    x1 = np.concatenate([cloud[1:], wind[1:], precipitation[1:]], axis=1)
    x2 = [np.expand_dims(humidity[1:-10], -1),
          np.expand_dims(pressure[1:-10], -1),
          np.expand_dims(temperature[1:-10], -1)]
    return x1, x2, y


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
    x_1, (x_2, x_3, x_4), target = load_data("../data")
    training_data = train_test_split((x_1, x_2, x_3, x_4), target)
    x1_train, x2_train, x3_train, x4_train, y_train, x1_test, x2_test, x3_test, x4_test, y_test = training_data
