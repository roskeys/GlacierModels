import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from utils import load_data, load_check_point

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

# load data
x, smb = load_data('data')
smb_df = pd.read_csv('./data/QAJUUTTAP_SERMIA_dm.csv')
year_min = smb_df['Year'].min()
year_range = [{'label': year_min + i, 'value': year_min + i} for i in range(len(smb))]
month_range = [str(i) for i in range(5, 13)] + [str(i) for i in range(1, 5)]
year_list = [year_min + i for i in range(len(smb))]
cloud = x[0][:, :12]
wind = x[0][:, 12:24]
precipitation = x[0][:, 24:36]
humidity = x[1]
pressure = x[2]
temperature = x[3]
height = min(humidity.shape[1], pressure.shape[1], temperature.shape[1])

# preprocessing
feature_1d = pd.DataFrame()
feature_2d = pd.DataFrame()
for y in range(len(smb)):
    df_1d = pd.DataFrame({
        'year': [y + year_min] * 12,
        'month': month_range,
        'cloud': cloud[y],
        'wind': wind[y],
        'precipitation': precipitation[y],
    })
    feature_1d = pd.concat([feature_1d, df_1d], axis=0)

models = []
for model_name in os.listdir('models'):
    for model_index, running_time in enumerate(os.listdir(os.path.join('models', model_name)), 1):
        for model_sub_index, checkpoint in enumerate(
                os.listdir(os.path.join('models', model_name, running_time, 'checkpoints')), 1):
            models.append({'label': model_name + '-' + str(model_index) + '-' + str(model_sub_index),
                           'value': os.path.join('models', model_name, running_time, 'checkpoints', checkpoint)})

app.layout = html.Div(children=[
    html.H1('Greenland Glacier surface mass balance model', style={'text-align': 'center'}),
    html.Div(children=[
        html.Div(children=[
            html.H4('Pretrained model selection', style={'width': '100%'}),
            dcc.Dropdown(
                options=models,
                id='model_selected',
                value=models[-1]['value'],
                style={'width': '100%'}
            ),
        ], style={'float': 'left', 'width': '30%'}),

        html.Div(children=[
            dcc.Graph(id='smb_compare'),
        ], style={
            'float': 'left', 'width': '60%'
        }),
    ], style={'height': '40%', 'width': '100%'}),

    html.Div(style={'clear': 'both'}),
    html.Div(children=[
        html.H4('Year'),
        dcc.Dropdown(
            options=year_range,
            id='year_range',
            value=year_range[-1]['value'], style={'width': '100%'}),
    ], style={'float': 'left', 'width': '15%'}),
    html.Div(children=[
        dcc.Graph(id='feature-graph-cloud', style={'float': 'left'}),
        dcc.Graph(id='feature-graph-wind', style={'float': 'left'}),
        dcc.Graph(id='feature-graph-precipitation', style={'float': 'left'}),
    ], style={'float': 'left', 'width': '40%', 'margin-right': '50px'}),
    html.Div(children=[
        dcc.Graph(id='feature-graph-humidity', style={'float': 'left'}),
        dcc.Graph(id='feature-graph-pressure', style={'float': 'left'}),
        dcc.Graph(id='feature-graph-temperature', style={'float': 'left'}),
    ], style={'float': 'left', 'width': '40%'}),
])


@app.callback(
    [
        Output(component_id='feature-graph-cloud', component_property='figure'),
        Output(component_id='feature-graph-wind', component_property='figure'),
        Output(component_id='feature-graph-precipitation', component_property='figure'),
        Output(component_id='feature-graph-humidity', component_property='figure'),
        Output(component_id='feature-graph-pressure', component_property='figure'),
        Output(component_id='feature-graph-temperature', component_property='figure'),
    ],
    [
        Input(component_id='year_range', component_property='value'),
    ]
)
def change_feature_plot(years):
    if isinstance(years, int):
        years = [years]
    cloud_plot, wind_plot, precipitation_plot, humidity_plot, pressure_plot, temperature_plot = [go.Figure() for _ in
                                                                                                 range(6)]
    for year in years:
        df = feature_1d[feature_1d['year'] == year]

        cloud_plot.add_trace(go.Scatter(x=df['month'], y=df['cloud'], mode='lines+markers'))
        cloud_plot.update_layout(title=f'{year}-cloud')
        cloud_plot.update_xaxes(type='category', categoryorder='array')

        wind_plot.add_trace(go.Scatter(x=df['month'], y=df['wind'], mode='lines+markers'))
        wind_plot.update_layout(title=f'{year}-wind')
        wind_plot.update_xaxes(type='category', categoryorder='array')

        precipitation_plot.add_trace(go.Scatter(x=df['month'], y=df['precipitation'], mode='lines+markers'))
        precipitation_plot.update_layout(title=f'{year}-precipitation')
        precipitation_plot.update_xaxes(type='category', categoryorder='array')

        humidity_plot.add_trace(go.Surface(x=np.arange(1, 13), y=np.arange(40) * 100 + 100,
                                           z=np.squeeze(humidity[year - year_min], -1)))
        humidity_plot.update_layout(title=f'{year}-humidity')

        pressure_plot.add_trace(go.Surface(x=np.arange(1, 13), y=np.arange(40) * 100 + 100,
                                           z=np.squeeze(pressure[year - year_min], -1)))
        pressure_plot.update_layout(title=f'{year}-pressure')

        temperature_plot.add_trace(go.Surface(x=np.arange(1, 13), y=np.arange(40) * 100 + 100,
                                              z=np.squeeze(temperature[year - year_min], -1)))
        temperature_plot.update_layout(title=f'{year}-temperature')
    cloud_plot.update_layout(autosize=True, margin=dict(t=30, b=20, l=20, r=20))
    wind_plot.update_layout(autosize=True, margin=dict(t=30, b=20, l=20, r=20))
    precipitation_plot.update_layout(autosize=True, margin=dict(t=30, b=20, l=20, r=20))
    humidity_plot.update_layout(autosize=True, margin=dict(t=30, b=20, l=20, r=20))
    pressure_plot.update_layout(autosize=True, margin=dict(t=30, b=20, l=20, r=20))
    temperature_plot.update_layout(autosize=True, margin=dict(t=30, b=20, l=20, r=20))
    return cloud_plot, wind_plot, precipitation_plot, humidity_plot, pressure_plot, temperature_plot


@app.callback(
    Output(component_id='smb_compare', component_property='figure'),
    Input(component_id='model_selected', component_property='value')
)
def change_comparison_plot(model_path):
    model = load_check_point(model_path)
    pred = model.predict(x)[:, 0]
    fig = go.Figure(data=[
        go.Scatter(x=np.array(year_list), y=pred, mode='lines+markers', name='Predicted'),
        go.Scatter(x=np.array(year_list), y=smb, mode='lines+markers', name='Actual')
    ])
    fig.update_layout(title='Comparison Between Predicted and Actual')
    fig.update_layout(autosize=True, margin=dict(t=50, b=20, l=20, r=20))
    return fig


if __name__ == '__main__':
    app.run_server(host='0.0.0.0')
