import dash
from dash import html, dcc, callback, Output, Input, State, Patch
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
import os

PATH = os.getcwd()
DICTSUBFOLDERS = [dict({'label': f.path, 'value': f.path}) for f in os.scandir(PATH) if f.is_dir()]
DICTSUBFOLDERS.append(dict({'label': PATH, 'value': PATH}))
DICTSUBFOLDERS.append(dict({'label': 'local paths are only available', 'value': 'local', 'disabled': True}))
CURTXT = [dict({'label': f.path.split('\\')[-1], 'value': f.path}) for f in os.scandir(PATH) if f.is_file() and f.path.split('.')[-1].lower() == 'txt']

# print(CURCSV)
# print('end')

dash.register_page(__name__, path='/analysis')
layout = html.Div([
    html.H3('Frame analysis'),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Frames"),
                dbc.CardBody([
                    dbc.InputGroup([
                        dbc.InputGroupText("Select folder"),
                        dbc.Select(id="loadPaths", options=DICTSUBFOLDERS),
                    ], size="sm"),
                    dbc.InputGroup([
                        dbc.InputGroupText("Current folder"),
                        dbc.Input(id="currentLoadPath", value=PATH.split('\\')[-1], disabled=True)#, style={'text-align': 'right'})
                    ], size="sm"),
                    dbc.Label("Choose frames", size="sm"),
                    dbc.Checklist(id='frames', options=CURTXT, style={'font-size': '8px'}),
                    dbc.Button("Open frames", id="openFrames", outline=True, color="primary", size="sm"),
                ]),
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dcc.Graph(id='plotFrames',
                          figure=go.Figure(),
                          config={"displayModeBar": False}
                ),
            ])
        ], width=9),
    ]),
    html.Br(),
])

#path selection
@callback(
Output('plotFrames', 'figure'),
    Output('currentLoadPath', 'value'),
    Output('frames', 'options'),
    Input('loadPaths', 'value'),
    prevent_initial_call=True
)
def pathSelection(path):
    fig = go.Figure()
    print(path)
    curtxt = [dict({'label': f.path.split('\\')[-1], 'value': f.path}) for f in os.scandir(path) if f.is_file() and f.path.split('.')[-1].lower() == 'txt']
    if curtxt:
        # print('there are csvs')
        curtxt.insert(0, dict({'label': 'add all', 'value': 'addall'}))
    print(path.split('\\')[-1])
    print(curtxt)
    return fig, path.split('\\')[-1], curtxt

#open frames
@callback(
    Output('plotFrames', 'figure', allow_duplicate=True),
    Output('frames', 'value', allow_duplicate=True),
    Input('openFrames', 'n_clicks'),
    State('frames', 'value'),
    State('frames', 'options'),
    State('loadPaths', 'value'),
    prevent_initial_call=True
)
def openFrames(n_clicks, value, options, path):
    print(f'number of clicks: {n_clicks}')
    fig = go.Figure()
    print(value)
    if value:
        if 'addall' in value: #data[0] == 'addall':
            value = [f.path for f in os.scandir(path) if f.is_file() and f.path.split('.')[-1].lower() == 'txt']
        for i, line in enumerate(value):
            arr = []
            with open(line, 'r') as file:
                for line2 in file:
                    arr.append([float(x) for x in line2.split()])
            # arr = np.genfromtxt(line, delimiter=',')
            fname = line.split('\\')[-1]
            print(arr[0])
            print(arr[1])
            print(arr[4])
            fig = fig.add_trace(go.Scatter(x=arr[0], y=arr[1], mode='markers', marker=dict(size=1, showscale=False), name=f'{i}: {fname}'))
            # data = value
    else:
        print('nothing to open')
        # data = [f.path for f in os.scandir(path) if f.is_file() and f.path.split('.')[-1].lower() == 'txt']

    return fig, value
