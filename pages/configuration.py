import dash
from dash import html, dcc, callback, Output, Input, State, Patch
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
import datetime as dt
import os
import scanner
# from scanner import connect, getInfo, getXZIExtended, makeFig

pointer, status = None, None
PATH = os.getcwd()
DICTSUBFOLDERS = [dict({'label': f.path, 'value': f.path}) for f in os.scandir(PATH) if f.is_dir()]
DICTSUBFOLDERS.append(dict({'label': 'local paths are only available', 'value': 'local', 'disabled': True}))
# print(DICTSUBFOLDERS)
#INTERVAL in msec 33.33ms/30FPS 16.66ms/60FPS
# INTERVAL = 17
INTERVAL = 33
# INTERVAL = 50
POINTS = 1280
#POINTS = 2560
START, STOP, RUNTIME, PROGTIME, INT = dt.datetime.now().timestamp()+1, 0, 0, 0, -1
# fig = go.Figure(data=go.Scatter(x=[], y=[], mode='markers', marker=dict(size=1, color='green', showscale=False)))
# figROI=go.Figure(data=go.Scatter(x=[scanner.MINX+scanner.DELTA,
#                                     scanner.MAXX-scanner.DELTA,
#                                     scanner.MAXX, scanner.MINX,
#                                     scanner.MINX+scanner.DELTA],
#                                  y=[scanner.MINZ,
#                                     scanner.MINZ,
#                                     scanner.MAXZ,
#                                     scanner.MAXZ,
#                                     scanner.MINZ],
#                                  mode='lines',
#                                  line=dict(color='red'),
#                                  ),
#                  )
traces = []
trace1 = go.Scatter(x=[scanner.MINX+scanner.DELTA, scanner.MAXX-scanner.DELTA, scanner.MAXX, scanner.MINX, scanner.MINX+scanner.DELTA],
                    y=[scanner.MINZ, scanner.MINZ, scanner.MAXZ, scanner.MAXZ, scanner.MINZ],
                    name='scanning area',
                    mode='lines',
                    line=dict(color='red', width=2)
                    )
trace2 = go.Scatter(x=[scanner.MINX, scanner.MAXX, scanner.MAXX, scanner.MINX, scanner.MINX],
                    y=[scanner.MINZ, scanner.MINZ, scanner.MAXZ, scanner.MAXZ, scanner.MINZ],
                    name='ROI',
                    mode='lines',
                    line=dict(color='green', width=1)
                    )
traces.append(trace1)
traces.append(trace2)
figROI=go.Figure()
figROI.add_trace(traces[0])
figROI.add_trace(traces[1])
figROI.update_xaxes(fixedrange=True)
figROI.update_yaxes(fixedrange=True)#, autorange='reversed')

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
                                dbc.Input(id="frequency", value=INTERVAL, placeholder="30", type="number"),
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
                                dbc.Button("Fix baseline", id="fixBaseline", outline=True, color="primary", size="sm"),
                                dbc.Input(id="baseline", value="0", type="number"),
                            ]),
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
                                dbc.ButtonGroup([
                                    dbc.Button("Aplay ROI", id="aplayROI", outline=True, color="primary", size="sm"),
                                    dbc.Button("Reset ROI", id="resetROI", outline=True, color="primary", size="sm")
                                ], size="sm")
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
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='plotROI',
                                      figure=figROI,
                                      config={"displayModeBar": False}
                                      # figure=go.Figure(data=go.Scatter(x=[scanner.MINX+scanner.DELTA, scanner.MAXX-scanner.DELTA, scanner.MAXX, scanner.MINX, scanner.MINX+scanner.DELTA],
                                      #                                  y=[scanner.MINZ, scanner.MINZ, scanner.MAXZ, scanner.MAXZ, scanner.MINZ],
                                      #                                  mode='lines',
                                      #                                  line=dict(color='red'))
                                      #                  )
                                      ),
                        ],
                        width=4),
                        dbc.Col([
                            dcc.Graph(id='plot',
                                      figure=go.Figure(),
                                      config={"displayModeBar": False}
                                      # figure=fig

                                      # figure=go.Figure(data=go.Scatter(x=[], y=[],
                                      #                                  mode='markers',
                                      #                                  marker=dict(size=1, color='green', showscale=False))
                                      #                  )
                                      ),
                        ],
                        width=8),
                    ]),
                    html.H5('Log info:'),
                    html.Output(id='logInformation'),
                    html.Br(),
                ])
            ]),
        ],
        width=8),
    ]),
    dcc.Interval('interval', interval=INTERVAL, n_intervals=0, max_intervals=0),
    dcc.Store(id='connectionStatus', data=0),
    dcc.Store(id='ROI'),
    dcc.Store(id='baselineStore'),
    dcc.Store(id='intensity'),
    dcc.Store(id='width')
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
    State('intensity', 'data'),
    State('width', 'data'),
    prevent_initial_call=True
)
def saveFrame(n_clicks, value, fig, intensity, width):
    currentTime = dt.datetime.now().strftime('%Hh_%Mm_%Ss')
    print(value)
    print(f'saved to {value}{currentTime}')
    # print(fig.get('data')[0]['x'])
    # print(fig.get('data')[0]['y'])
    # define NumPy array
    data = np.array([np.around(fig.get('data')[0]['x'], 4),
                     np.around(fig.get('data')[0]['y'], 4),
                     np.around(intensity, 4),
                     np.around(width, 4)])
    # print(data)
    # export array to CSV file
    np.savetxt(f'{value}\\{currentTime}.csv', data, fmt="%1.2f", delimiter=",")

    return f'saved to {value}{currentTime}.csv'


#update frequency
@callback(
    Output('logInformation', 'children', allow_duplicate=True),
    Output('interval', 'interval'),
    Input('updateFrequency', 'n_clicks'),
    State('frequency', 'value'),
    prevent_initial_call=True
)
def updateFrequency(n_clicks, value):
    global INTERVAL
    INTERVAL = int(1000/value)
    print(f'update frequency in ms: {INTERVAL}')
    return f'update frequency in ms: {INTERVAL}', INTERVAL


#connection
@callback(
    Output('logInformation', 'children', allow_duplicate=True),
    Output('connectionStatus', 'data'),
    Output('status', 'value'),
    Input('connect', 'n_clicks'),
    State('connectionStatus', 'data'),
    prevent_initial_call=True
)
def connection(n_clicks, data):
    strStatus = "OFF"
    global pointer, status
    if not status:
        pointer, status = scanner.connect(scanner.lib)
        scanner.writeToSensor(pointer, scanner.lib, "SetAcquisitionStop\r")
        res = scanner.resetDllFiFo(scanner.lib, pointer)
        count = 0
        while True:
            dataLength, bufX, bufZ, bufIntensity, bufSignalWidth = scanner.getXZIExtended(scanner.lib, pointer)
            if dataLength == -1:
                print(f'Buffer is cleaned {dataLength}({count}) first time')
                break
            count += 1
        scanner.writeToSensor(pointer, scanner.lib, "SetAcquisitionStart\r")
        if status: strStatus = "ON"
    else: strStatus = "ON"

    return f"connection status: {status}", status, strStatus


#update intervals with ROI
@callback(
    Output('plot', 'figure', allow_duplicate=True),
    Output('logInformation', 'children', allow_duplicate=True),
    Output('intensity', 'data'),
    Output('width', 'data'),
    Input('interval', 'n_intervals'),
    Input('ROI', 'data'),
    Input('baselineStore', 'data'),
    State('connectionStatus', 'data'),
    # State('plot', 'figure'),
    prevent_initial_call=True,
)
def update_intervals(n_intervals, data, baseline, s):
    global START, STOP, RUNTIME, PROGTIME, INT
    # global fig
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
    arr1, arr2, min, arr3, arr4 = generateData(n_intervals, s)
    # fig = go.Figure()
    fig = go.Figure(data=go.Scattergl(x=arr1, y=arr2,
                                      mode='markers',
                                      marker=dict(size=1, color='green', showscale=False)
                                      ))
    fig.add_trace(go.Scattergl(x=[minX, maxX], y=[baseline, baseline], mode='lines', line_dash='dash', name='baseline'))
    fig.add_trace(go.Scattergl(x=[min[0]], y=[min[1]], mode="markers+text", text=[round(min[1], 2)], textposition="bottom center"))
    fig.update_xaxes(range=[minX, maxX], tickangle=90, zeroline=False)#dtick=0.2,
    fig.update_yaxes(range=[minZ, maxZ], zeroline=False)#, autorange='reversed')#dtick=0.2,
    fig.update_layout({'xaxis': {'scaleanchor': 'y'}}, showlegend=False)

    # fig.add_trace(go.Scattergl(x=[max[0]], y=[max[1]], mode="markers", marker=dict(size=1, color='green', showscale=False)))

    # with open('data.txt', 'a') as f:
    #     f.write(f'{min[0]}, {min[1]};\n')

    return fig, f'{round(min[0], 2)}, {round(min[1], 2)}', arr3, arr4


def generateData(step, flag):
    global pointer
    if not flag:
        # x0 = 0
        # x1 = np.pi
        # arr1 = np.linspace(x0, x1, POINTS)
        # s = 2*np.pi/POINTS
        # arr2 = arr1*0
        # arr2 = -1*abs(np.sin((arr1 + 10*s*step)*0.5) + np.cos((arr1 + 10*s*step)*2))
        arr1 = np.linspace(-31,31, 500)
        x = arr1 + step
        # arr2 = 116 + (100 * np.sin(0.1 * x) + 50 * np.cos(0.2 * x) + 20 * np.sin(0.3 * x) - 10 * np.cos(0.4 * x))
        arr2 = np.clip(121 - 45 * np.sin(0.1 * x) - 20 * np.cos(0.2 * x) - 10 * np.sin(0.3 * x) + 5 * np.cos(0.4 * x) - 8 * np.sin(0.6 * x) + 0.5 * x, 80, 158)
        arr3 = np.empty(POINTS)
        arr3[:] = step
        arr4 = np.empty(POINTS)
        arr4[:] = step*10
    else:
        if scanner.resetDllFiFo(scanner.lib, pointer): print("Reset DLL FiFo UNsuccessful!")
        # print(f'Reset DLL FiFo result: {scanner.resetDllFiFo(scanner.lib, pointer)}')
        # count = 0
        # while True:
        #     dataLength, bufX, bufZ, bufIntensity = scanner.getXZIExtended(scanner.lib, pointer)
        #     if dataLength == -1:
        #         print(f'Buffer is cleaned {dataLength}({count})')
        #         break
        #     count += 1
        dataLength, arrX, arrZ, bufIntensity, bufSignalWidth = scanner.getXZIExtended(scanner.lib, pointer)
        arr1, arr2, arr3, arr4 = scanner.transformData(arrX, arrZ, bufIntensity, bufSignalWidth)
    # print(f'len {len(arr1), len(arr2)}')
    # min = []
    if len(arr2):
        minIndex = np.argmin(arr2)
        min = [arr1[minIndex], arr2[minIndex]]
    else:
        min = [0, 0]
    # print(max)

    return arr1, arr2, min, arr3, arr4


#zoom by plot
@callback(
    Output('logInformation', 'children', allow_duplicate=True),
    Output('ROI', 'data'),
    Output('minX', 'value'),
    Output('maxX', 'value'),
    Output('minZ', 'value'),
    Output('maxZ', 'value'),
    Output('plotROI', 'figure'),
    Input('plot', 'relayoutData'),
    # State('plotROI', 'figure'),
    prevent_initial_call=True
)
def updateByPlot(data):
    global figROI, traces
    if ('xaxis.range[0]' in data) and ('yaxis.range[0]' in data) == True:
        minX = round(data['xaxis.range[0]'], 2)
        maxX = round(data['xaxis.range[1]'], 2)
        minZ = round(data['yaxis.range[0]'], 2)
        maxZ = round(data['yaxis.range[1]'], 2)
    else:
        minX, maxX, minZ, maxZ = scanner.MINX, scanner.MAXX, scanner.MINZ, scanner.MAXZ
    # trace2 = go.Scatter()
    # print(figure['data'])
    # figROI.update_traces(traces)
    # figROI.add_trace(traces[0])
    # figROI.add_trace(traces[1])
    # print(figROI['data'][1]['x'])
    command = f'SetROI1_mm={round(minX, 0)},{round(minZ, 0)},{round(maxX, 0)},{round(maxZ, 0)}\r'
    print(f'command: {command}')
    scanner.writeToSensor(pointer, scanner.lib, command)
    figROI['data'][1]['x'] = (minX, maxX, maxX, minX, minX)
    figROI['data'][1]['y'] = (minZ, minZ, maxZ, maxZ, minZ)
    # figROI.add_trace(go.Scatter(x=figure['data'][0]['x'], y=figure['data'][0]['y']))
    return 'plot zoomed', data, minX, maxX, minZ, maxZ, figROI


#zoom by ROI
@callback(
    Output('logInformation', 'children', allow_duplicate=True),
    Output('ROI', 'data', allow_duplicate=True),
    Output('plotROI', 'figure', allow_duplicate=True),
    Input('aplayROI', 'n_clicks'),
    State('minX', 'value'),
    State('maxX', 'value'),
    State('minZ', 'value'),
    State('maxZ', 'value'),
    prevent_initial_call=True
)
def updateByROI(n_clicks, minX, maxX, minZ, maxZ):
    global figROI
    #print(f"from zoom {minX, maxX, minZ, maxZ}")
    data = {}
    data['xaxis.range[0]'] = minX
    data['xaxis.range[1]'] = maxX
    data['yaxis.range[0]'] = minZ
    data['yaxis.range[1]'] = maxZ
    command = f'SetROI1_mm={round(minX, 0)},{round(minZ, 0)},{round(maxX, 0)},{round(maxZ, 0)}\r'
    print(f'command: {command}')
    scanner.writeToSensor(pointer, scanner.lib, command)
    figROI['data'][1]['x'] = (minX, maxX, maxX, minX, minX)
    figROI['data'][1]['y'] = (minZ, minZ, maxZ, maxZ, minZ)
    return 'ROI zoomed', data, figROI


#reset ROI
@callback(
    Output('logInformation', 'children', allow_duplicate=True),
    Output('ROI', 'data', allow_duplicate=True),
    Output('plotROI', 'figure', allow_duplicate=True),
    Output('minX', 'value', allow_duplicate=True),
    Output('maxX', 'value', allow_duplicate=True),
    Output('minZ', 'value', allow_duplicate=True),
    Output('maxZ', 'value', allow_duplicate=True),
    Input('resetROI', 'n_clicks'),
    prevent_initial_call=True
)
def resetROI(n_clicks):
    global figROI
    data = {}
    data['xaxis.range[0]'] = scanner.MINX
    data['xaxis.range[1]'] = scanner.MAXX
    data['yaxis.range[0]'] = scanner.MINZ
    data['yaxis.range[1]'] = scanner.MAXZ
    figROI['data'][1]['x'] = (scanner.MINX, scanner.MAXX, scanner.MAXX, scanner.MINX, scanner.MINX)
    figROI['data'][1]['y'] = (scanner.MINZ, scanner.MINZ, scanner.MAXZ, scanner.MAXZ, scanner.MINZ)
    command = f'SetROI1_mm={scanner.MINX},{scanner.MINZ},{scanner.MAXX},{scanner.MAXZ}\r'
    print(f'command: {command}')
    scanner.writeToSensor(pointer, scanner.lib, command)
    return 'ROI reset', data, figROI, scanner.MINX, scanner.MAXX, scanner.MINZ, scanner.MAXZ

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
    return n_intervals, 36000000#303


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


#data generation
@callback(
    Output('plot', 'figure'),
    Input('generation', 'n_clicks'),
    prevent_initial_call=True,
)
def updateData(n_clicks):
    global fig
    optimizationType = 'opengl'
    arr1, arr2, max, arr3, arr4 = generateData(0, 0)
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
    arr1, arr2, max = generateData(n_clicks, 0)
    fig = go.Figure(data=go.Scatter(x=arr1, y=arr2,
                                    mode='markers',
                                    marker=dict(size=1, color='green', showscale=False)
                                    ))
    return fig