import dash
from dash import html, dcc, callback, Output, Input, State, Patch
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
import datetime as dt
import os
import scanner

PATH = os.getcwd()
DICTSUBFOLDERS = [dict({'label': f.path, 'value': f.path}) for f in os.scandir(PATH) if f.is_dir()]
DICTSUBFOLDERS.append(dict({'label': 'local paths are only available', 'value': 'local', 'disabled': True}))
# print(DICTSUBFOLDERS)
#INTERVAL in msec 33.33ms/30FPS 16.66ms/60FPS
# INTERVAL = 16.66
INTERVAL = 33
# INTERVAL = 60
POINTS = 1280
#POINTS = 2560
START, STOP, RUNTIME, PROGTIME, INT = dt.datetime.now().timestamp()+1, 0, 0, 0, -1
fig = go.Figure(data=go.Scatter(x=[], y=[],
                                                       mode='markers',
                                                       marker=dict(size=1, color='green', showscale=False))
                                       )
dash.register_page(__name__, path='/')

layout = html.Div([
    html.H3('Sensor configuration'),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Settings"),
                dbc.CardBody([
                    dbc.Card([
                        dbc.CardHeader("Connection status"),
                        dbc.CardBody([
                            dbc.InputGroup([
                                dbc.InputGroupText("IP"),
                                dbc.Input(id="ip", value="192.168.255.8", placeholder="192.168.255.8"),
                                dbc.InputGroupText("Status"),
                                dbc.Input(id="status", value="OFF", disabled=True)
                            ], size="sm"),
                            html.Br(),
                            dbc.InputGroup([
                                dbc.InputGroupText("Port"),
                                dbc.Input(id="port", value="32001", placeholder="32001", type="number"),
                                dbc.InputGroupText("Frequency"),
                                dbc.Input(id="frequency", value="30", placeholder="30", type="number"),
                            ], size="sm"),
                            html.Br(),
                            dbc.ButtonGroup([
                                dbc.Button("Connect", id="connect", outline=True, color="primary"),
                                dbc.Button("Disconnect", id="disconnect", outline=True, color="primary"),
                                dbc.Button("Update frequency", id="updateFrequency", outline=True, color="primary")
                            ], size="sm")
                        ])
                    ]),
                    dbc.Card([
                        dbc.CardHeader("Geometry settings"),
                        dbc.CardBody([
                            dbc.InputGroup([
                                dbc.Button("Fix baseline", id="fixBaseline", outline=True, color="primary"),
                                dbc.Input(id="baseline", value="0", type="number"),
                            ], size="sm"),
                            html.Br(),
                            dbc.InputGroup([
                                dbc.InputGroup([
                                    dbc.InputGroupText("ROI min X"),
                                    dbc.Input(id="minX", value=scanner.MINX, type="number"),
                                    dbc.InputGroupText("ROI max X"),
                                    dbc.Input(id="maxX", value=scanner.MAXX, type="number"),
                                ], size="sm"),
                                dbc.InputGroup([
                                    dbc.InputGroupText("ROI min Z"),
                                    dbc.Input(id="minZ", value=scanner.MINZ, type="number"),
                                    dbc.InputGroupText("ROI max Z"),
                                    dbc.Input(id="maxZ", value=scanner.MAXZ, type="number")
                                ], size="sm"),
                                dbc.Button("Aplay ROI", id="aplayROI", outline=True, color="primary", size="sm")
                            ]),
                        ])
                    ]),
                    dbc.Card([
                        dbc.CardHeader("Saving information"),
                        dbc.CardBody([
                            dbc.InputGroup([
                                dbc.InputGroupText("Select folder"),
                                dbc.Select(id="paths", options=DICTSUBFOLDERS),
                            ], size="sm"),
                            dbc.InputGroup([
                                dbc.InputGroupText("Current folder"),
                                dbc.Input(id="currentPath", value=PATH, disabled=True)
                            ], size="sm"),
                            dbc.Button("Save frame", id="saveFrame", outline=True, color="primary", size="sm"),
                        ])
                    ]),
                    dbc.Card([
                        dbc.CardHeader("Scanner simulation"),
                        dbc.CardBody([
                            dbc.ButtonGroup([
                                dbc.Button('Data generation', id='generation', outline=True, color="primary"),
                                dbc.Button('Next step', id='next', outline=True, color="primary")
                            ], size="sm"),
                            html.Br(),
                            dbc.ButtonGroup([
                                dbc.Button('Start  scanning', id='start', outline=True, color="primary"),
                                dbc.Button('Pause  scanning', id='pause', outline=True, color="primary"),
                                dbc.Button('Stop   scanning', id='stop', outline=True, color="primary")
                            ], size="sm")
                        ])
                    ]),
                ])
            ]),
        ],
        width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Realtime scanning graph"),
                dbc.CardBody([
                    # html.H4('Realtime scanning graph'),
                    dcc.Graph(id='plot',
                              figure=fig
                              # figure=go.Figure(data=go.Scatter(x=[], y=[],
                              #                                  mode='markers',
                              #                                  marker=dict(size=1, color='green', showscale=False))
                              #                  )
                              ),
                    html.H5('Log info:'),
                    html.Output(id='logInformation'),
                    html.Br(),
                ])
            ]),
        ],
        width=8),
    ]),
    dcc.Interval('interval', interval=INTERVAL, n_intervals=0, max_intervals=0),
    dcc.Store(id='ROI'),
    dcc.Store(id='baselineStore')
])


#path selection
@callback(
    Output('currentPath', 'value'),
    Input('paths', 'value'),
    prevent_initial_call=True
)
def pathSelection(path):
    # print(path)
    return path


#save frame
@callback(
    Output('logInformation', 'children', allow_duplicate=True),
    Input('saveFrame', 'n_clicks'),
    State('currentPath', 'value'),
    State('plot', 'figure'),
    prevent_initial_call=True
)
def saveFrame(n_clicks, value, fig):
    currentTime = dt.datetime.now().strftime('%Hh_%Mm_%Ss')
    print(value)
    print(f'saved to {value}{currentTime}')
    # print(fig.get('data')[0]['x'])
    # print(fig.get('data')[0]['y'])
    # define NumPy array
    data = np.array([fig.get('data')[0]['x'], fig.get('data')[0]['y']])
    # export array to CSV file
    np.savetxt(f'{value}\\{currentTime}.csv', data, delimiter=",")
    return f'saved to {value}{currentTime}.csv'


#update frequency
@callback(
    Output('logInformation', 'children', allow_duplicate=True),
    Input('updateFrequency', 'n_clicks'),
    State('frequency', 'value'),
    prevent_initial_call=True
)
def updateFrequency(n_clicks, value):
    global INTERVAL
    INTERVAL = int(1000/value)
    print(f'update frequency: {INTERVAL}')
    return f'update frequency: {INTERVAL}'

#update intervals with ROI
@callback(
    Output('plot', 'figure', allow_duplicate=True),
    Output('logInformation', 'children'),
    Input('interval', 'n_intervals'),
    Input('ROI', 'data'),
    Input('baselineStore', 'data'),
    prevent_initial_call=True,
)
def update_intervals(n_intervals, data, baseline):
    global START, STOP, RUNTIME, PROGTIME, INT
    INT += 1
    if not (RUNTIME):
        msg = f'({POINTS} points): count {INT} (progtime in sec: {INTERVAL * INT / 1000} and realtime: {int(dt.datetime.now().timestamp() - START)})'
    else:
        msg = f'({POINTS} points): count {INT} (progtime in sec: {PROGTIME} and realtime: {int(RUNTIME)})'
    if ('xaxis.range[0]' in data) and ('yaxis.range[0]' in data) == True:
        minX = data['xaxis.range[0]']
        maxX = data['xaxis.range[1]']
        minZ = data['yaxis.range[0]']
        maxZ = data['yaxis.range[1]']
    else:
        minX, maxX, minZ, maxZ = scanner.MINX, scanner.MAXX, scanner.MINZ, scanner.MAXZ
    arr1, arr2, max = generateData(n_intervals)
    # fig = go.Figure()
    fig = go.Figure(data=go.Scattergl(x=arr1, y=arr2,
                                      mode='markers',
                                      marker=dict(size=1, color='green', showscale=False)
                                      ))
    fig.add_trace(go.Scattergl(x=[minX, maxX], y=[baseline, baseline], mode='lines', line_dash='dash', name='baseline'))
    fig.add_trace(go.Scattergl(x=[max[0]], y=[max[1]], mode="markers+text", text=[round(max[1], 2)], textposition="bottom center"))
    fig.update_xaxes(range=[minX, maxX], dtick=0.2, tickangle=90, zeroline=False)
    fig.update_yaxes(range=[minZ, maxZ], dtick=0.2, zeroline=False)
    fig.update_layout({'xaxis': {'scaleanchor': 'y'}}, showlegend=False)
    # fig.update_layout(xaxis={'scaleanchor': 'y'}, grid_ygap=0.1, grid_xgap=0.1)

    return fig, msg


#zoom by plot
@callback(
    Output('logInformation', 'children', allow_duplicate=True),
    Output('ROI', 'data'),
    Output('minX', 'value'),
    Output('maxX', 'value'),
    Output('minZ', 'value'),
    Output('maxZ', 'value'),
    Input('plot', 'relayoutData'),
    prevent_initial_call=True
)
def updateByPlot(data):
    #print(f"from zoom {relayoutData}")
    if ('xaxis.range[0]' in data) and ('yaxis.range[0]' in data) == True:
        minX = round(data['xaxis.range[0]'], 2)
        maxX = round(data['xaxis.range[1]'], 2)
        minZ = round(data['yaxis.range[0]'], 2)
        maxZ = round(data['yaxis.range[1]'], 2)
    else:
        minX, maxX, minZ, maxZ = scanner.MINX, scanner.MAXX, scanner.MINZ, scanner.MAXZ
    return 'plot zoomed', data, minX, maxX, minZ, maxZ


#zoom by ROI
@callback(
    Output('logInformation', 'children', allow_duplicate=True),
    Output('ROI', 'data', allow_duplicate=True),
    Input('aplayROI', 'n_clicks'),
    State('minX', 'value'),
    State('maxX', 'value'),
    State('minZ', 'value'),
    State('maxZ', 'value'),
    prevent_initial_call=True
)
def updateByROI(n_clicks, minX, maxX, minZ, maxZ):
    #print(f"from zoom {minX, maxX, minZ, maxZ}")
    data = {}
    data['xaxis.range[0]'] = minX
    data['xaxis.range[1]'] = maxX
    data['yaxis.range[0]'] = minZ
    data['yaxis.range[1]'] = maxZ

    return 'ROI zoomed', data


#fix baseline
@callback(
    Output('baselineStore', 'data'),
    Input('fixBaseline', 'n_clicks'),
    State('baseline', 'value'),
    prevent_initial_call=True
)
def fixBaseline(n_clicks, value):
    print(f'baseline: {value}')
    return value


#data generation
@callback(
    Output('plot', 'figure'),
    Input('generation', 'n_clicks'),
    prevent_initial_call=True,
)
def updateData(n_clicks):
    global fig
    optimizationType = 'opengl'
    arr1, arr2, max = generateData(0)
    if optimizationType == 'standard':
        fig = go.Figure(data=go.Scatter(x=arr1, y=arr2,
                                        mode='markers',
                                        marker=dict(size=1, color='green', showscale=False)
                                        ))
    elif optimizationType == 'opengl':
        fig = go.Figure(data=go.Scattergl(x=arr1, y=arr2,
                                        mode='markers',
                                        marker=dict(size=1, color='green', showscale=False)
                                        ))
    return fig


#next step
@callback(
    Output('plot', 'figure', allow_duplicate=True),
    Input('next', 'n_clicks'),
    prevent_initial_call=True,
)
def updateNextData(n_clicks):
    arr1, arr2, max = generateData(n_clicks)
    fig = go.Figure(data=go.Scatter(x=arr1, y=arr2,
                                    mode='markers',
                                    marker=dict(size=1, color='green', showscale=False)
                                    ))
    return fig

def generateData(step):
    x0 = 0
    x1 = 2*np.pi
    arr1 = np.linspace(x0, x1, POINTS)
    s = 2*np.pi/POINTS
    arr2 = np.sin((arr1 + 10*s*step)*0.5) + np.cos((arr1 + 10*s*step)*2)
    maxIndex = np.argmax(arr2)
    max = [arr1[maxIndex], arr2[maxIndex]]

    # print(s*step, arr1, arr2)
    return arr1, arr2, max


"""
#timer
@callback(
    Output('countInfo', 'children', allow_duplicate=True),
    Output('plot', 'figure', allow_duplicate=True),
    Input('interval', 'n_intervals'),
    Input('optimizationType', 'value'),
    Input('patched', 'value'),
    prevent_initial_call=True,
)
def update_intervals(n_intervals, typeValue, patchedValue):
    global INT
    INT += 1
    arr1, arr2 = generateData(n_intervals)
    if typeValue == 'standard':
        fig = go.Figure(data=go.Scatter(x=arr1, y=arr2,
                                        mode='markers',
                                        marker=dict(size=1, color='green', showscale=False)
                                        ))
    elif typeValue == 'opengl':
        if patchedValue == ['patched figure']:
            fig = Patch()
            fig["data"][0]["x"] = arr1
            fig["data"][0]["y"] = arr2
        else:
            fig = go.Figure(data=go.Scattergl(x=arr1, y=arr2,
                                            mode='markers',
                                            marker=dict(size=1, color='green', showscale=False)
                                            ))
    return n_intervals, fig
"""


#start
@callback(
    Output('interval', 'n_intervals'),
    Output('interval', 'max_intervals'),
    Input('start', 'n_clicks'),
    State('interval', 'n_intervals'),
    prevent_initial_call=True,
)
def start(n_clicks, n_intervals):
    global START, STOP, RUNTIME, PROGTIME, INT
    STOP = 0
    RUNTIME, PROGTIME, INT = 0, 0, 0
    START = dt.datetime.now().timestamp()
    return n_intervals, 36000000


#pause
@callback(
    Output('interval', 'max_intervals', allow_duplicate=True),
    Input('pause', 'n_clicks'),
    State('interval', 'n_intervals'),
    prevent_initial_call=True,
)
def pause(n_clicks, n_intervals):
    global START, STOP, RUNTIME, PROGTIME, INT
    STOP = dt.datetime.now().timestamp()
    RUNTIME = STOP - START
    PROGTIME = INTERVAL*n_intervals/1000
    START = STOP
    INT = n_intervals
    return 0


#stop
@callback(
    Output('interval', 'max_intervals', allow_duplicate=True),
    Output('interval', 'n_intervals', allow_duplicate=True),
    Input('stop', 'n_clicks'),
    State('interval', 'n_intervals'),
    prevent_initial_call=True,
)
def stop(n_clicks, n_intervals):
    global START, STOP, RUNTIME, PROGTIME, INT
    STOP = dt.datetime.now().timestamp()
    PROGTIME = INTERVAL * n_intervals / 1000
    RUNTIME = STOP - START
    INT = n_intervals
    return 0, 0

"""
#timer output
@callback(
    Output('logInfo', 'children'),
    Input('interval', 'n_intervals'),
    Input('optimizationType', 'value'),
    Input('patched', 'value'),
    prevent_initial_call=True,
)
def logInfo(n_intervals, typeValue, patchedValue):
    global START, STOP, RUNTIME, PROGTIME, INT
    if not(RUNTIME):
        return f'with {typeValue} method with {patchedValue} ({POINTS} points):{INT} (progtime in sec: {INTERVAL*INT/1000} and realtime: {int(dt.datetime.now().timestamp() - START)})'
    else:
        return f'with {typeValue} method with {patchedValue} ({POINTS} points):{INT} (progtime in sec: {PROGTIME} and realtime: {int(RUNTIME)})'
    #RUNTIME = dt.datetime.now().timestamp() - START
    #return f'with {value} method ({POINTS} points):{n_intervals} (progtime in sec: {INTERVAL*n_intervals/1000} and realtime: {int(RUNTIME)})'
    #return f'with {value} method ({POINTS} points):{n_intervals} (progtime in sec: {PROGTIME} and realtime: {int(RUNTIME)})'
"""