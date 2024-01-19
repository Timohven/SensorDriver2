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
CURCSV = [dict({'label': f.path, 'value': f.path}) for f in os.scandir(PATH) if f.is_file() and f.path.split('.')[-1].lower() == 'csv']

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
                    dbc.Checklist(id='frames', options=CURCSV, style={'font-size': '8px'}),
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
    ])
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
    curcsv = [dict({'label': f.path, 'value': f.path}) for f in os.scandir(path) if f.is_file() and f.path.split('.')[-1].lower() == 'csv']
    if curcsv:
        # print('there are csvs')
        curcsv.insert(0, dict({'label': 'add all', 'value': 'addall'}))
    print(path.split('\\')[-1])
    print(curcsv)
    return fig, path.split('\\')[-1], curcsv

#open frames
@callback(
    Output('plotFrames', 'figure', allow_duplicate=True),
    Input('openFrames', 'n_clicks'),
    State('frames', 'value'),
    State('loadPaths', 'value'),
    prevent_initial_call=True
)
def openFrames(n_clicks, data, path):
    fig = go.Figure()
    print(data)
    if data:
        if 'addall' in data: #data[0] == 'addall':
            data = [f.path for f in os.scandir(path) if f.is_file() and f.path.split('.')[-1].lower() == 'csv']
        for line in data:
            arr = np.genfromtxt(line, delimiter=',')
            # print(arr)
            fig = fig.add_trace(go.Scatter(x=arr[0], y=arr[1], mode='markers', marker=dict(size=1, showscale=False), name=line.split('\\')[-1]))
    return fig