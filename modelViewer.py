import os
import pickle
import re

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import dash
import numpy as np
import plotly.graph_objects as go
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from utils import load_check_point

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

saved_model_base_path = "saved_models"
get_base_path = re.compile("(.*?)saved_checkpoints")
models = []
model_folders = os.listdir(saved_model_base_path)

if "loss" in model_folders:
    model_folders.remove("loss")
if "PredictedvsActual" in model_folders:
    model_folders.remove("PredictedvsActual")

for model_name in model_folders:
    for model_index, running_time in enumerate(os.listdir(os.path.join(saved_model_base_path, model_name)), 1):
        base_path = os.path.join(saved_model_base_path, model_name, running_time)
        checkpoint_list = os.listdir(os.path.join(base_path, "saved_checkpoints"))
        if len(checkpoint_list) > 0:
            models.append({
                'label': model_name + '-' + running_time,
                'value': os.path.join(base_path, 'saved_checkpoints', checkpoint_list[-1])
            })

app.layout = html.Div(children=[
    html.H1('Greenland Glacier surface mass balance model', style={'text-align': 'center'}),
    html.Div(children=[
        html.Div(children=[
            html.H4('Pretrained model selection', style={'width': '100%'}),
            dcc.Dropdown(
                options=models,
                id='model_selected',
                value=models[0]['value'],
                style={'width': '100%'}
            ),
        ], style={'float': 'left', 'width': '20%'}),

        html.Div(children=[
            dcc.Graph(id='smb_compare'),
        ], style={
            'float': 'left', 'width': '80%'
        }),
    ], style={'height': '100%', 'width': '100%', "padding": "20px"}),
])


@app.callback(
    Output(component_id='smb_compare', component_property='figure'),
    Input(component_id='model_selected', component_property='value')
)
def change_comparison_plot(model_path):
    model_base_path = get_base_path.search(model_path).groups()[0]
    with open(os.path.join(model_base_path, "data.pickle"), 'rb') as f:
        x, smb = pickle.load(f)
    data_size = len(smb)
    model = load_check_point(model_path)
    pred = model.predict(x)[:, 0]
    fig = go.Figure(data=[
        go.Scatter(x=np.arange(data_size), y=pred, mode='lines+markers', name='Predicted'),
        go.Scatter(x=np.arange(data_size), y=smb, mode='lines+markers', name='Actual')
    ])
    fig.update_layout(title='Comparison Between Predicted and Actual')
    fig.update_layout(autosize=True, margin=dict(t=50, b=20, l=20, r=20))
    return fig


if __name__ == '__main__':
    app.run_server(host="0.0.0.0")
