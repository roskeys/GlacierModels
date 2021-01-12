import os
import re
import numpy as np
import pandas as pd

index_matcher = re.compile(".*?/(\d{5})/?")


def load_1d_data(dataframe):
    feature_name = dataframe.columns[-1]
    if "year" not in dataframe.columns or "month" not in dataframe.columns:
        if "YEAR" in dataframe.columns:
            dataframe = dataframe.rename(columns={"YEAR": "year"})
        if "MONTH" in dataframe.columns:
            dataframe = dataframe.rename(columns={"MONTH": "month"})
        if dataframe.columns[1] == '0' and dataframe.columns[2] == '1':
            dataframe = dataframe.rename(columns={'0': "year", '1': "month"}, errors="raise")
    year_range = dataframe["year"].unique()
    year_range.sort()
    output = pd.DataFrame()
    for year in year_range[:-1]:
        data = {"year": year + 1}
        df_current_year = dataframe[dataframe["year"] == year]
        for i in range(5, 13):
            data[i] = df_current_year.loc[df_current_year["month"] == i][feature_name].values
        next_year = year + 1
        df_next_year = dataframe[dataframe["year"] == next_year]
        for i in range(1, 5):
            data[i] = df_next_year.loc[df_next_year["month"] == i][feature_name].values
        try:
            df = pd.DataFrame(data)
        except ValueError as e:
            if str(e) == "arrays must all be same length":
                continue
        output = pd.concat([output, df], ignore_index=True)
    return output


def load_2d_data(dataframe):
    height = dataframe.shape[0]
    year_range = [int(year) for year in dataframe.columns.str.split("_").str[0].unique()[2:]]
    year_range.sort()
    output_data = pd.DataFrame()
    for year in year_range[:-2]:
        data = {"year": np.array([year + 1] * height)}
        df_current_year = dataframe.filter(regex=f"^{year}", axis=1)
        for i in range(5, 13):
            data[i] = df_current_year.filter(regex=f"_{i}$", axis=1).values.squeeze()
        next_year = year + 1
        df_next_year = dataframe.filter(regex=f"^{next_year}", axis=1)
        for i in range(1, 5):
            data[i] = df_next_year.filter(regex=f"_{i}$", axis=1).values.squeeze()
        data['height'] = np.arange(0, height) * 100
        try:
            df = pd.DataFrame(data)
        except ValueError as e:
            if str(e) == "arrays must all be same length":
                continue
            elif str(e) == "Data must be 1-dimensional":
                continue
        output_data = pd.concat([output_data, df], ignore_index=True)
    return output_data


def load_smb(name, df):
    df = df[df["NAME"] == name]
    columns = df.filter(regex="\d{4}\.5").columns
    columns = [int(year) for year in columns]
    sorted(columns)
    smb = pd.DataFrame()
    for year in columns:
        smb = pd.concat([smb, pd.DataFrame({"year": year, "SMB": df[year + 0.5].values})])
    return smb


def load_1d_array(df, year_range):
    df = df[df['year'].isin(year_range)]
    return df[[i for i in range(1, 13)]].values


def load_2d_array(df, year_range):
    df = df[df['year'].isin(year_range)]
    data = []
    for year in df["year"].unique():
        data.append(df[df["year"] == year][[i for i in range(1, 13)]].values)
    return np.array(data)


def load_smb_array(df, year_range):
    df = df[df['year'].isin(year_range)]
    return df["SMB"].values


def get_common_year_range(year_range, df):
    return year_range.intersection(set(df["year"]))


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


def concatenate_data(x1, y1, x2, y2):
    x_concatenated = []
    if isinstance(x1, list) or isinstance(x1, tuple):
        for x_part1, x_part2 in zip(x1, x2):
            x_concatenated.append(np.concatenate([x_part1, x_part2], axis=0))
    else:
        x_concatenated = np.concatenate([x1, x2], axis=0)
    y_concatenated = np.concatenate([y1, y2])
    return x_concatenated, y_concatenated


def load_data_by_cluster(glacier_assignment_path, glacier_name, centroid_to_igra_map, igra_base_path, dmi_base_path,
                         smb_path):
    smb_df = load_smb(glacier_name, pd.read_excel(smb_path))
    glacier_assignment = pd.read_csv(glacier_assignment_path)
    central = glacier_assignment[glacier_assignment['NAME'] == glacier_name]['Central'].values[0]
    igra_name = centroid_to_igra_map[central]
    humidity_df = load_2d_data(pd.read_csv(os.path.join(igra_base_path, igra_name, "CalHum_std.csv")))
    pressure_df = load_2d_data(pd.read_csv(os.path.join(igra_base_path, igra_name, "Pressure.csv")))
    temperature_df = load_2d_data(pd.read_csv(os.path.join(igra_base_path, igra_name, "Temp.csv")))
    cloud_df = load_1d_data(pd.read_csv(os.path.join(dmi_base_path, str(central), f"mean_cloud_{central}.csv")))
    wind_df = load_1d_data(pd.read_csv(os.path.join(dmi_base_path, str(central), f"mean_wind_{central}.csv")))
    precipitation_df = load_1d_data(
        pd.read_csv(os.path.join(dmi_base_path, str(central), f"monthly_total_precipitation_{central}.csv")))
    dataframes = [smb_df, humidity_df, pressure_df, temperature_df, cloud_df, wind_df, precipitation_df]
    common_year_range = set(smb_df["year"])
    for df in dataframes[1:]:
        common_year_range = get_common_year_range(common_year_range, df)
    smb_array = load_smb_array(smb_df, common_year_range)
    x_data_set = [
        load_1d_array(cloud_df, common_year_range),
        load_1d_array(precipitation_df, common_year_range),
        load_1d_array(wind_df, common_year_range),
        load_2d_array(humidity_df, common_year_range),
        load_2d_array(pressure_df, common_year_range),
        load_2d_array(temperature_df, common_year_range)
    ]
    return x_data_set, smb_array


if __name__ == "__main__":
    centroid_map = {5: 'AASIAAT(EGEDESMINDE)', 1: 'DANMARKSHAVN', 2: 'ITTOQQORTOORMIIT', 4: 'MITTARFIK_NARSARSUAQ',
                    3: 'TASIILAQ(AMMASSALIK)'}  # 'PITUFFIK',

    x_all, y_all = load_data_by_cluster("../data/glacier_assignment.csv", "ILORLIIT-SERMINNGUAQ", centroid_map,
                                        "../data/IGRA Archieves/", "../data/newDMI8_data",
                                        "../data/smb_mass_change.xlsx")

    for x in x_all:
        print(x.shape)
    print(y_all.shape)
