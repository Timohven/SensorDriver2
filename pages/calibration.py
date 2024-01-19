import dash
from dash import html, dcc, callback, Output, Input, State, Patch
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
import os
import math
from utilities import matrices

PATH = os.getcwd()
DICTSUBFOLDERS = [dict({'label': f.path, 'value': f.path}) for f in os.scandir(PATH) if f.is_dir()]
DICTSUBFOLDERS.append(dict({'label': PATH, 'value': PATH}))
DICTSUBFOLDERS.append(dict({'label': 'local paths are only available', 'value': 'local', 'disabled': True}))
CURCSV = [dict({'label': f.path, 'value': f.path}) for f in os.scandir(PATH) if f.is_file() and f.path.split('.')[-1].lower() == 'csv']

dash.register_page(__name__, path='/calibration')
layout = html.Div([
    html.H3('Calibration'),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Frames"),
                dbc.CardBody([
                    dbc.InputGroup([
                        dbc.InputGroupText("Select folder"),
                        dbc.Select(id="loadPaths2", options=DICTSUBFOLDERS),
                    ], size="sm"),
                    dbc.InputGroup([
                        dbc.InputGroupText("Current folder"),
                        dbc.Input(id="currentLoadPath2", value=PATH.split('\\')[-1], disabled=True)#, style={'text-align': 'right'})
                    ], size="sm"),
                    dbc.Label("Choose frames", size="sm"),
                    dbc.Checklist(id='frames2', options=CURCSV, style={'font-size': '8px'}),
                    dbc.Button("Open frames", id="openFrames2", outline=True, color="primary", size="sm"),
                ]),
            ]),
            # dbc.InputGroup([
                dbc.Button("Calculate parallel", id="calcParallel", outline=True, color="primary", size="sm"),
                html.Span("Non yet calculated", id="calculated", style={"verticalAlign": "middle"}),
            # ], size="sm"),
        ], width=3),
        dbc.Col([
            dbc.Card([
                dcc.Graph(id='plotFrames2',
                          figure=go.Figure(),
                          # config={"displayModeBar": False}
                ),
            ])
        ], width=9),
    ]),
    dcc.Store(id='currentFrame'),
])

#path selection
@callback(
Output('plotFrames2', 'figure'),
    Output('currentLoadPath2', 'value'),
    Output('frames2', 'options'),
    Input('loadPaths2', 'value'),
    prevent_initial_call=True
)
def pathSelection(path):
    fig = go.Figure()
    curcsv = [dict({'label': f.path, 'value': f.path}) for f in os.scandir(path) if f.is_file() and f.path.split('.')[-1].lower() == 'csv']
    print(path.split('\\')[-1])
    return fig, path.split('\\')[-1], curcsv

#open frames
@callback(
    Output('plotFrames2', 'figure', allow_duplicate=True),
    Output('currentFrame', 'data'),
    Input('openFrames2', 'n_clicks'),
    State('frames2', 'value'),
    prevent_initial_call=True
)
def openFrames(n_clicks, data):
    fig = go.Figure()
    arr = None
    # print(data)
    if data:
        for line in data:
            arr = np.genfromtxt(line, delimiter=',')
            # print(arr)
            fig = fig.add_trace(go.Scatter(x=arr[0], y=arr[1], mode='markers', marker=dict(size=1, showscale=False), name=line.split('\\')[-1]))

    return fig, arr

#calculate parallel
@callback(
    Output('calculated', 'children'),
    Output('plotFrames2', 'figure', allow_duplicate=True),
    Input('calcParallel', 'n_clicks'),
    State('currentFrame', 'data'),
    State('plotFrames2', 'figure'),
    prevent_initial_call=True
)
def calcParallel(n_clicks, data, fig):
    N = 3 #number of sequential values
    DELTA = 10 #hight difference value
    lim1 = sum(data[1][0:N])/N - DELTA
    lim2 = sum(data[1][-N:])/N - DELTA
    # print(lim1, lim2)
    [first, i] = next([el, i] for i, el in enumerate(data[1][:]) if el < lim1)
    # print(first, data[0][i], i)
    [last, j] = next([el, i] for i, el in enumerate(data[1][::-1]) if el < lim2)
    # print(last, data[0][-j-1], -j-1)
    range1 = data[0][i] - data[0][-j-1]
    newFig = go.Figure(fig)
    dist1 = math.sqrt((data[0][-j - 1] - data[0][i]) ** 2 + (last - first) ** 2)
    newFig.add_trace(go.Scatter(x=[data[0][i], data[0][-j-1]], y=[first, last],
                                mode='markers', marker=dict(size=3, color='red', showscale=False),
                                name=f'range1={range1}, dist1={dist1}'))
    range2 = data[0][i-1] - data[0][-j]
    dist2 = math.sqrt((data[0][-j]-data[0][i-1])**2+(data[1][-j]-data[1][i-1])**2)
    newFig.add_trace(go.Scatter(x=[data[0][i-1], data[0][-j]], y=[data[1][i-1], data[1][-j]],
                                mode='markers', marker=dict(size=3, color='red', showscale=False),
                                name=f'range2={range2}, dist2={dist2}'))
    # print(data[0][::2])
    # arr = data[0, 0,,2]
    x = np.array(data[0][i:-j-1])
    x_np = x.reshape(-1, 1)
    vector_1 = np.ones((x_np.shape[0], 1))
    x_np = np.hstack((vector_1, x_np))
    y = np.array(data[1][i:-j-1])
    y_np = y.reshape(-1, 1)
    # print(x_np.shape)
    # print(x_np)
    # print(y_np.shape)
    # print(y_np)
    ab_np = matrices.matrix_equation(x_np, y_np)
    ab_np = np.around(ab_np, 4)
    a = ab_np[0][0]
    b = ab_np[1][0]
    newFig.add_trace(go.Scatter(x=[data[0][i], data[0][-j-1]], y=[a+data[0][i]*b, a+data[0][-j-1]*b],
                                mode='lines', line_dash='dash', name=f'{a}+x*{b}'))
    TCP = [553126,209910,59905,150,-27,923818]
    RU = [557013,236623,54609]
    LU = [519210,238346,55042]
    dist = math.sqrt((RU[0]/1000-LU[0]/1000)**2 + (RU[1]/1000-LU[1]/1000)**2)
    print(dist)
    return f'Calculated {n_clicks}', newFig

