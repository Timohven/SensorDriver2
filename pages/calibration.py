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
CURTXT = [dict({'label': f.path, 'value': f.path}) for f in os.scandir(PATH) if f.is_file() and f.path.split('.')[-1].lower() == 'txt']

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
                    dbc.Checklist(id='frames2', options=CURTXT, style={'font-size': '8px'}),
                    dbc.Button("Open frames", id="openFrames2", outline=True, color="primary", size="sm"),
                ]),
            ]),
            html.Br(),
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
            ]),
            dbc.Card([
                dcc.Graph(id='fig3D',
                          figure=go.Figure(go.Scatter3d()),
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
    Output('frames2', 'value'),
    Input('loadPaths2', 'value'),
    prevent_initial_call=True
)
def pathSelection(path):
    fig = go.Figure()
    curtxt = [dict({'label': f.path, 'value': f.path}) for f in os.scandir(path) if f.is_file() and f.path.split('.')[-1].lower() == 'txt']
    print(path.split('\\')[-1])
    return fig, path.split('\\')[-1], curtxt, []

#open frames
@callback(
    Output('plotFrames2', 'figure', allow_duplicate=True),
    Output('currentFrame', 'data'),
    Input('openFrames2', 'n_clicks'),
    State('frames2', 'value'),
    prevent_initial_call=True
)
def openFrames(n_clicks, val):
    fig = go.Figure()
    arr = None
    print(f'selected frame: {val}')
    value = []
    if val:
        value.append(val[-1])

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
        fig = fig.add_trace(
            go.Scatter(x=arr[0], y=arr[1], mode='markers', marker=dict(size=1, showscale=False), name=f'{i}: {fname}'))
        # data = value
    # if data:
    #     for line in data:
    #         arr = np.genfromtxt(line, delimiter=',')
    #         # print(arr)
    #         fig = fig.add_trace(go.Scatter(x=arr[0], y=arr[1], mode='markers', marker=dict(size=1, showscale=False), name=line.split('\\')[-1]))

    return fig, arr

#calculate parallel
@callback(
    Output('calculated', 'children'),
    Output('plotFrames2', 'figure', allow_duplicate=True),
    Output('fig3D', 'figure'),
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
    B = [data[0][i], data[1][i]]
    print(f'B: {B}')
    # print(first, data[0][i], i)
    [last, j] = next([el, i] for i, el in enumerate(data[1][::-1]) if el < lim2)
    C = [data[0][-j-1], data[1][-j-1]]
    print(f'C: {C}')
    A = [data[0][i-1], data[1][i-1]]
    print(f'A: {A}')
    D = [data[0][-j], data[1][-j]]
    print(f'D: {D}')
    # print(D, type(D))
    # print(last, data[0][-j-1], -j-1)

    hCalc1 = B[1] - A[1]
    hCalc2 = C[1] - D[1]
    print(f'hCalc1, hCalc2 = {hCalc1, hCalc2}')
    rangeBCx = B[0] - C[0]
    BC = np.linalg.norm(np.array(B) - np.array(C))
    newFig = go.Figure(fig)
    newFig.add_trace(go.Scatter(x=[B[0], C[0]], y=[B[1], C[1]],
                                mode='markers', marker=dict(size=3, color='red', showscale=False),
                                name=f'range1={rangeBCx}, distBC={BC}'))
    rangeADx = A[0] - D[0]
    AD = np.linalg.norm(np.array(A) - np.array(D))
    newFig.add_trace(go.Scatter(x=[A[0], D[0]], y=[A[1], D[1]],
                                mode='markers', marker=dict(size=3, color='red', showscale=False),
                                name=f'range2={rangeADx}, dist2={AD}'))
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
    newFig.add_trace(go.Scatter(x=[B[0], C[0]], y=[a+B[0]*b, a+C[0]*b],
                                mode='lines', line_dash='dash', name=f'{a}+x*{b}'))
    # TCP = [553126,209910,59905,150,-27,923818]
    # RU = [557013,236623,54609]
    # LU = [519210,238346,55042]
    # dist = math.sqrt((RU[0]/1000-LU[0]/1000)**2 + (RU[1]/1000-LU[1]/1000)**2)
    # print(dist)
    newFig.add_trace(go.Scatter(x=[A[0], B[0]], y=[A[1], B[1]],
                                mode='lines', line_dash='dash', name=f'AB'))
    newFig.add_trace(go.Scatter(x=[D[0], C[0]], y=[D[1], C[1]],
                                mode='lines', line_dash='dash', name=f'DC'))

    # a1x+b1y=c1; a2x+b2y=c2
    # x-y=0; -x-y=-1
    # -kx+y=b
    k1, b1 = matrices.line(*A, *B)
    k2, b2 = matrices.line(*D, *C)
    mM = np.matrix([[-k1, 1], [-k2, 1]])
    mC = np.matrix([b1, b2])
    mO = mM.I.dot(mC.T)
    O = np.asarray(mO).reshape(-1).tolist()
    print(f'O: {O}')

    kAD = (A[1]-D[1])/(A[0]-D[0])
    bAD = D[1]-kAD*D[0]
    centerLaserAD = [0, bAD]

    kBC = (B[1] - C[1]) / (B[0] - C[0])
    bBC = C[1] - kBC * C[0]
    centerLaserBC = [0, bBC]

    OcenterLaserAD = np.linalg.norm(np.array(O) - np.array(centerLaserAD))
    print(f'OcenterLaserAD: {OcenterLaserAD}')

    OcenterLaserBC = np.linalg.norm(np.array(O) - np.array(centerLaserBC))
    print(f'OcenterLaserBC: {OcenterLaserBC}')

    newFig.add_trace(go.Scatter(x=[O[0], centerLaserAD[0], centerLaserBC[0]], y=[O[1], centerLaserAD[1], centerLaserBC[1]],
                                mode='markers', marker=dict(size=3, color='red', showscale=False),
                                name=f'O, centerLaserAD, centerLaserBC'))
    newFig.update_layout({'xaxis': {'scaleanchor': 'y'}})#, showlegend=False)

    #3D lines
    h = 39
    AB = np.linalg.norm(np.array(A) - np.array(B))
    AO = np.linalg.norm(np.array(A) - np.array(O))
    sinAlfa = h/AB
    H = sinAlfa * np.linalg.norm(np.array(A)-np.array(O))
    print(f'sin Alfa: {sinAlfa}')
    print(f'AB: {AB}')
    print(f'AO: {AO}')
    print(f'H: {H}')


    DC = np.linalg.norm(np.array(D) - np.array(C))
    DO = np.linalg.norm(np.array(D) - np.array(O))
    sinBeta = h / DC
    H1 = sinBeta * np.linalg.norm(np.array(D) - np.array(O))
    print(f'sin Beta: {sinBeta}')
    print(f'DC: {DC}')
    print(f'DO: {DO}')
    print(f'H1: {H1}')

    AB = np.linalg.norm(np.array(A) - np.array(B))
    AB1 = AB * np.cos(np.arcsin(sinAlfa))
    print(f'AB1: {AB1}')

    print(f'BC: {BC}')

    AO = np.linalg.norm(np.array(A) - np.array(O))
    OO1 = AO * np.sin(np.arcsin(sinAlfa))
    print(f'OO1: {OO1}')

    Gamma = np.arcsin(H/OcenterLaserAD)
    print(f'Gamma: {Gamma}')
    print(f'geometric deviation of the laser beam from the Z axis UF1: {90 - np.degrees(Gamma)} degrees')
    centerLaserAD = np.array([0, OcenterLaserAD*np.cos(Gamma), 0])
    print(f'new centerLaserAD: {centerLaserAD}')
    bNewAD = centerLaserAD[1] - kAD * centerLaserAD[0]

    centerLaserBC = np.array([0, OcenterLaserBC * np.cos(Gamma), 0])
    print(f'new centerLaserBC: {centerLaserBC}')
    bNewBC = centerLaserBC[1] - kBC * centerLaserBC[0]

    A1 = [A[0], kAD * A[0] + bNewAD]
    D1 = [D[0], kAD * D[0] + bNewAD]

    B1 = [B[0], kBC * B[0] + bNewBC]
    C1 = [C[0], kBC * C[0] + bNewBC]

    kA1D1 = (A1[1] - D1[1]) / (A1[0] - D1[0])
    bA1D1 = D1[1] - kA1D1 * D1[0]
    centerLaserA1D1 = [0, bA1D1]
    A1centerLaserA1D1 = np.linalg.norm(np.array(A1) - np.array(centerLaserA1D1))
    print(f'A1centerLaserA1D1: {A1centerLaserA1D1}')
    Teta = np.arccos(A1[0]/A1centerLaserA1D1)
    print(f'Teta: {Teta}')
    print(f'geometric deviation of the laser beam from the X axis UF1: {np.degrees(Teta)} degrees')

    newFig.add_trace(go.Scatter(x=[A1[0], D1[0], B1[0], C1[0], centerLaserA1D1[0]],
                                y=[A1[1], D1[1], B1[1], C1[1], centerLaserA1D1[1]],
                                mode='markers', marker=dict(size=3, color='black', showscale=False),
                                name=f'A1, D1, B1, C1, centerLaserA1D1'))


    # 3D points
    O1 = np.array([0, 0, 0])
    O = np.array([0, 0, H])
    A1.append(0)
    A1 = np.array(A1)
    D1.append(0)
    D1 = np.array(D1)

    A1O1 = np.linalg.norm(A1 - O1)
    print(f'A1O1: {A1O1}')

    B1.append(0)
    B1 = np.array(B1)
    C1.append(0)
    C1 = np.array(C1)

    B = B1.copy()
    B[2] = h
    C = C1.copy()
    C[2] = h

    fig3D = go.Figure(go.Scatter3d())
    #main points
    fig3D.add_trace(go.Scatter3d(x=[O1[0], O[0], centerLaserAD[0], A1[0], D1[0], B1[0], C1[0], B[0], C[0]],
                                 y=[O1[1], O[1], centerLaserAD[1], A1[1], D1[1], B1[1], C1[1], B[1], C[1]],
                                 z=[O1[2], O[2], centerLaserAD[2], A1[2], D1[2], B1[2], C1[2], B[2], C[2]],
                                 name='main points',
                                 mode='markers',
                                 text=['O1', 'O', 'centerLaserAD', 'A1', 'D1', 'B1', 'C1', 'B', 'C'],
                                 marker=dict(size=2, color='green')
                                 )
                    )
    #vector
    fig3D.add_trace(go.Scatter3d(x=[O[0], centerLaserAD[0]],
                                 y=[O[1], centerLaserAD[1]],
                                 z=[O[2], centerLaserAD[2]],
                                 name='Z axis of the laser',
                                 line=dict(color='red', width=3),
                                 marker=dict(symbol='circle-open', size=3, color='red')
                                 )
                    )

    # fig3D.add_trace(go.Scatter3d(x=[O[0], centerLaserAD[0]],
    #                              y=[O[1], centerLaserAD[1]],
    #                              z=[O[2], centerLaserAD[2]],
    #                              mode='lines+markers',
    #                              marker=dict(symbol=["arrow-right"], size=3, color='green', angleref="previous",)
    #                              )
    #                 )

    #section
    fig3D.add_trace(go.Mesh3d(x=[B1[0], C1[0], C[0], B[0]],
                              y=[B1[1], C1[1], C[1], B[1]],
                              z=[B1[2], C1[2], C[2], B[2]],
                              i=[0, 0],
                              j=[1, 2],
                              k=[2, 3],
                              opacity=0.5,
                              name='experimental standard cross section'
                              )
                    )

    #laser beam
    fig3D.add_trace(go.Mesh3d(x=[O[0], B[0], C[0], B1[0], C1[0], A1[0], D1[0]],
                              y=[O[1], B[1], C[1], B1[1], C1[1], A1[1], D1[1]],
                              z=[O[2], B[2], C[2], B1[2], C1[2], A1[2], D1[2]],
                              i=[0],
                              j=[5],
                              k=[6],
                              opacity=0.3,
                              color='blue',
                              name='laser beam'
                              )
                    )
    # fig3D.add_trace(go.Scatter3d(x=[A[0], D[0], B[0], C[0], O[0]],
    #                              y=[A[1], D[1], B[1], C[1], O[1]],
    #                              z=[0, 0, h, h, H],
    #                              mode='markers',
    #                              marker=dict(size=3, color='green')
    #                              )
    #                 )
    # fig3D.update_traces(text=['A', 'D', 'B', 'C', 'O'], selector=dict(type='scatter3d'))

    # fig3D.add_trace(go.Scatter3d(x=[B[0], C[0], C[0], B[0], B[0]],
    #                              y=[B[1], C[1], C[1], B[1], B[1]],
    #                              z=[h, h, 0, 0, h],
    #                              mode='lines',
    #                              line=dict(width=3)
    #                              )
    #                 )

    # fig3D.add_trace(go.Scatter3d(x=[A[0], O[0], D[0], A[0]],
    #                              y=[A[1], O[1], D[1], A[1]],
    #                              z=[0, H, 0, 0],
    #                              mode='lines',
    #                              line=dict(width=1)
    #                              )
    #                 )

    # fig3D.update_layout(scene_camera=dict(eye=dict(x=0., y=0., z=2.5),
    #                                       up=dict(x=0., y=0., z=1),
    #                                       center=dict(x=0., y=0., z=0.)))
    # fig3D.update_layout(width=700, height=700, scene_camera_eye_z=0.8)

    return f'dX: {np.degrees(Teta)}, dZ: {90 - np.degrees(Gamma)}', newFig, fig3D

