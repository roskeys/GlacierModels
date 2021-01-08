import pandas as pd
import numpy as np


def load_1d_data(dataframe):
    feature_name = dataframe.columns[-1]
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
        output = pd.concat([output, pd.DataFrame(data)], ignore_index=True)
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
            record = df_current_year.filter(regex=f"_{i}$", axis=1)
            data[i] = record.values.squeeze()
        next_year = year + 1
        df_next_year = dataframe.filter(regex=f"^{next_year}", axis=1)
        for i in range(1, 5):
            record = df_next_year.filter(regex=f"_{i}$", axis=1)
            data[i] = record.values.squeeze()
        data['height'] = np.arange(0, height) * 100
        output_data = pd.concat([output_data, pd.DataFrame(data)], ignore_index=True)
    return output_data


if __name__ == '__main__':
    load_1d_data(pd.read_csv("MITTARFIK NARSARSUAQ/mean_cloud.csv")).to_csv("../MITTARFIK NARSARSUAQ/mean_cloud.csv")
    load_1d_data(pd.read_csv("MITTARFIK NARSARSUAQ/mean_wind.csv")).to_csv("../MITTARFIK NARSARSUAQ/mean_wind.csv")
    load_1d_data(pd.read_csv("MITTARFIK NARSARSUAQ/monthly_total_precipitation.csv")).to_csv(
        "../MITTARFIK NARSARSUAQ/mean_precipitation.csv")
    load_2d_data(pd.read_csv("MITTARFIK NARSARSUAQ/Temp_fill.csv")).to_csv("../MITTARFIK NARSARSUAQ/mean_temperature.csv")
    load_2d_data(pd.read_csv("MITTARFIK NARSARSUAQ/CalHum_std_fill.csv.csv")).to_csv("../MITTARFIK NARSARSUAQ/mean_humidity.csv")
    load_2d_data(pd.read_csv("MITTARFIK NARSARSUAQ/Pressure_fill.csv")).to_csv("../MITTARFIK NARSARSUAQ/mean_pressure.csv")
    smb = pd.read_csv("MITTARFIK NARSARSUAQ/QAJUUTTAP_SERMIA_dm.csv")
    smb.columns = ['year', 'SMB']
    smb.to_csv("../MITTARFIK NARSARSUAQ/smb.csv")
