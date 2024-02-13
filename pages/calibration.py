import dash
from dash import html, dcc, callback, Output, Input, State, Patch
import dash_bootstrap_components as dbc
import dash_ag_grid as dag
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
# import math
from utilities import matrices
import objects

PATH = os.getcwd()
DICTSUBFOLDERS = [dict({'label': f.path, 'value': f.path}) for f in os.scandir(PATH) if f.is_dir()]
DICTSUBFOLDERS.append(dict({'label': PATH, 'value': PATH}))
DICTSUBFOLDERS.append(dict({'label': 'local paths are only available', 'value': 'local', 'disabled': True}))
CURTXT = [dict({'label': f.path.split('\\')[-1], 'value': f.path}) for f in os.scandir(PATH) if f.is_file() and f.path.split('.')[-1].lower() == 'txt']

data = []
# df = pd.DataFrame.from_dict(data)
df = pd.DataFrame(data, columns=['frame',
                                 'UF1 Ry',
                                 'UF1 Rz',
                                 'Teta(Z and UF1)',
                                 'Gamma(X and UF1)',
                                 'h calculated',
                                 'top edge',
                                 'errO',
                                 'errOx',
                                 'errOy',
                                 'errA',
                                 'errD'
                                 ])



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
                    dbc.Button("Save report", id="saveReport", outline=True, color="primary", size="sm"),
                ]),
            ]),
            html.Br(),
            # dbc.InputGroup([
                dbc.Button("Calculate projection", id="calcParallel", outline=True, color="primary", size="sm"),
                html.Span("Non yet calculated", id="calculated", style={"verticalAlign": "middle"}),
                html.Br(),
                dbc.Button("Restore surface", id="restSurf", outline=True, color="primary", size="sm"),
                html.Br(),
                dbc.Button("Calculate coordinates", id="calcCoord", outline=True, color="primary", size="sm"),
                html.Span("Non yet calculated", id="coordinates", style={"verticalAlign": "middle"}),
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
            ]),
            dbc.Card([
                dag.AgGrid(
                    id="resultTable",
                    columnSize="sizeToFit",
                    columnDefs=[{'field': columnName} for columnName in df.columns],
                    defaultColDef={'resizable': True, 'sortable': True, 'filter': True},
                    rowData=df.to_dict("records"),
                    csvExportParams={
                        "fileName": "report.csv",
                    })
            ]),
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
    Output('resultTable', 'rowData'),
    Input('loadPaths2', 'value'),
    prevent_initial_call=True
)
def pathSelection(path):
    if len(df): df.drop(df.index[:], inplace=True)

    fig = go.Figure()
    curtxt = [dict({'label': f.path.split('\\')[-1], 'value': f.path}) for f in os.scandir(path) if f.is_file() and f.path.split('.')[-1].lower() == 'txt']
    print(path.split('\\')[-1])
    return fig, path.split('\\')[-1], curtxt, [], []

#open frames
@callback(
    Output('plotFrames2', 'figure', allow_duplicate=True),
    Output('currentFrame', 'data'),
    Input('openFrames2', 'n_clicks'),
    State('frames2', 'value'),
    prevent_initial_call=True
)
def openFrames(n_clicks, val):
    # fig = go.Figure()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
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
        print(f'Robot coord: {arr[4]}')
        # intensity bar
        intensityTrace = go.Bar(x=arr[0], y=arr[2], name='intensity', marker_color='blue', opacity=0.7)
        fig.add_trace(intensityTrace, secondary_y=True)
        fig = fig.add_trace(
            go.Scatter(x=arr[0], y=arr[1], mode='markers', marker=dict(size=1, showscale=False), name=f'{i}: {fname}'))
        # data = value
    # if data:
    #     for line in data:
    #         arr = np.genfromtxt(line, delimiter=',')
    #         # print(arr)
    #         fig = fig.add_trace(go.Scatter(x=arr[0], y=arr[1], mode='markers', marker=dict(size=1, showscale=False), name=line.split('\\')[-1]))

    return fig, arr


#save report
@callback(
    Output('resultTable', 'exportDataAsCsv'),
    Input('saveReport', 'n_clicks'),
    prevent_initial_call=True
)
def saveReport(n_clicks):
    df.to_excel('report.xlsx')#, engine='xlsxwriter')
    if n_clicks:
        return True
    return False


#calculate projection
@callback(
    Output('calculated', 'children'),
    Output('plotFrames2', 'figure', allow_duplicate=True),
    Output('fig3D', 'figure'),
    Output('resultTable', 'rowData', allow_duplicate=True),
    Input('calcParallel', 'n_clicks'),
    State('currentFrame', 'data'),
    State('plotFrames2', 'figure'),
    State('frames2', 'value'),
    prevent_initial_call=True
)
def calcProjection(n_clicks, data, fig, currentFrame):
    frame = objects.Frame2D(data[0], data[1])
    # print(frame)
    # approximation of a face using a straight line
    a, b = frame.faceApproximation(data[0], data[1])
    B = frame.approxB
    C = frame.approxC
    # B = frame.B
    # C = frame.C

    c, d = frame.shadowApproximation(data[0], data[1])
    A = frame.approxA
    D = frame.approxD

    #change A and D received from scanner to the calculated one
    # frame.calcShadow()
    # A = frame.calcA
    # D = frame.calcD

    errorA = frame.errorA
    errorD = frame.errorD

    hCalc1 = A[1] - B[1]
    hCalc2 = D[1] - C[1]
    print(f'hCalc1, hCalc2 = {hCalc1, hCalc2}')
    rangeBCx = B[0] - C[0]
    BC = np.linalg.norm(np.array(B) - np.array(C))
    newFig = go.Figure(fig)
    newFig.add_trace(go.Scatter(x=[B[0], C[0]], y=[B[1], C[1]],
                                mode='markers', marker=dict(size=3, color='red', showscale=False),
                                name=f'range1={rangeBCx}, distBC={BC}'))
    rangeADx = A[0] - D[0]
    AD = np.linalg.norm(np.array(A) - np.array(D))

    # finding the point of intersection of lines AB and DC
    calcO = matrices.linesIntersection(A, B, D, C)
    O = calcO
    print(f'calcO: {calcO}')

    errorO = np.linalg.norm(np.array([0, 0]) - np.array(calcO))
    errorOx = 0 - calcO[0]
    errorOy = 0 - calcO[1]

    O = calcO

    newFig.add_trace(go.Scatter(x=[A[0], D[0]], y=[A[1], D[1]],
                                mode='markers', marker=dict(size=3, color='red', showscale=False),
                                name=f'range2={rangeADx}, distAD={AD}'))

    newFig.add_trace(go.Scatter(x=[B[0]+10, C[0]-10], y=[a+(B[0]+10)*b, a+(C[0]-10)*b],
                                mode='lines', line=dict(color='black', width=1), line_dash='dash', name=f'BC: y={a}+x*{b}'))
    newFig.add_trace(go.Scatter(x=[A[0] + 10, D[0] - 10], y=[c + (A[0] + 10) * d, c + (D[0] - 10) * d],
                                mode='lines', line=dict(color='black', width=1), line_dash='dash', name=f'AD: y={c}+x*{d}'))

    newFig.add_trace(go.Scatter(x=[A[0], O[0]], y=[A[1], O[1]],
                                mode='lines', line=dict(color='red', width=1), line_dash='dash', name=f'ABO'))
    newFig.add_trace(go.Scatter(x=[D[0], O[0]], y=[D[1], O[1]],
                                mode='lines', line=dict(color='red', width=1), line_dash='dash', name=f'DCO'))

    #y = k*x + b
    #y = (yB-yA)/(xB-xA)*x - ((yB-yA)/(xB-xA)*xA-yA)
    kAD, bAD = matrices.line(*A, *D)
    # kAD = (A[1]-D[1])/(A[0]-D[0])
    # bAD = D[1]-kAD*D[0]
    centerLaserAD = [0, bAD]

    kBC, bBC = matrices.line(*B, *C)
    # kBC = (B[1] - C[1]) / (B[0] - C[0])
    # bBC = C[1] - kBC * C[0]
    centerLaserBC = [0, bBC]

    OcenterLaserAD = np.linalg.norm(np.array(O) - np.array(centerLaserAD))
    print(f'OcenterLaserAD: {OcenterLaserAD}')

    OcenterLaserBC = np.linalg.norm(np.array(O) - np.array(centerLaserBC))
    print(f'OcenterLaserBC: {OcenterLaserBC}')

    newFig.add_trace(go.Scatter(x=[O[0], centerLaserAD[0], centerLaserBC[0]], y=[O[1], centerLaserAD[1], centerLaserBC[1]],
                                mode='markers', marker=dict(size=3, color='red', showscale=False),
                                name=f'calcO, centerLaserAD, centerLaserBC',
                                text=['calcO', 'centerLaserAD', 'centerLaserBC']))
    newFig.add_trace(go.Scatter(x=[O[0], centerLaserAD[0]], y=[O[1], centerLaserAD[1]],
                                mode='lines', line=dict(color='black', width=1), line_dash='dash',
                                name=f'OcenterLaserAD'))
    newFig.update_layout({'xaxis': {'scaleanchor': 'y'}})#, showlegend=False)

    #3D angles
    h = 39.19
    # h = 30.0
    AB = np.linalg.norm(np.array(A) - np.array(B))
    AO = np.linalg.norm(np.array(A) - np.array(O))
    sinAlfa = h / AB
    alfaH = sinAlfa * AO
    alfa = np.arcsin(sinAlfa)
    print(f'AB, AO, Alfa: {AB, AO, np.degrees(alfa)}')

    DC = np.linalg.norm(np.array(D) - np.array(C))
    DO = np.linalg.norm(np.array(D) - np.array(O))
    sinBeta = h / DC
    betaH = sinBeta * DO
    beta = np.arcsin(sinBeta)
    print(f'DC, DO, Beta: {DC, DO, np.degrees(beta)}')

    sinGamma = h / np.linalg.norm(np.array(centerLaserBC) - np.array(centerLaserAD))
    np.linalg.norm(np.array(centerLaserBC) - np.array(centerLaserAD))
    # print(f'control: {np.linalg.norm(np.array(centerLaserBC) - np.array(centerLaserAD))}')
    print(f'sinGamma={sinGamma}')
    gammaH = sinGamma * OcenterLaserAD
    gamma = np.arcsin(sinGamma)
    print(f'Gamma: {np.degrees(gamma)}')
    print(f'alfaH, betaH, gammaH: {alfaH, betaH, gammaH}')

    #calculation projections
    Op = O.copy()
    OOp = gammaH
    print(f'OO1: {OOp}')
    print(f'geometric deviation of the laser beam from the Z axis UF1: {90 - np.degrees(gamma)} degrees')

    centerLaserApDpOld = [0, Op[1] + OcenterLaserAD * np.cos(gamma)]
    print(f'projection of centerLaserAD: {centerLaserApDpOld}')
    bApDp = centerLaserApDpOld[1]

    centerLaserBpCp = [0, Op[1] + OcenterLaserBC * np.cos(gamma)]
    print(f'projection of centerLaserBC: {centerLaserBpCp}')
    bBpCp = centerLaserBpCp[1]

    Bp = [B[0], kBC * B[0] + bBpCp]
    Cp = [C[0], kBC * C[0] + bBpCp]

    # Bp = [B[0], B[1] * np.cos(alfa)]
    # Cp = [C[0], C[1] * np.cos(beta)]

    # Ap = [A[0], kAD * A[0] + bApDp]
    # Dp = [D[0], kAD * D[0] + bApDp]
    # Ap = [A[0], A[1] * np.cos(gamma)]
    # Dp = [D[0], D[1] * np.cos(gamma)
    kOBp, bOBp = matrices.line(*O, *Bp)
    kOCp, bOCp = matrices.line(*O, *Cp)

    Ap = [A[0], kOBp * A[0] + bOBp]
    Dp = [D[0], kOCp * D[0] + bOCp]
    k, b = matrices.line(*Ap, *Dp)
    centerLaserApDp = [0, b]

    # kA1D1 = (A1[1] - D1[1]) / (A1[0] - D1[0])
    # bA1D1 = D1[1] - kA1D1 * D1[0]
    # centerLaserA1D1 = [0, bA1D1]
    # kB1C1 = (B1[1] - C1[1]) / (B1[0] - C1[0])
    # bB1C1 = C1[1] - kB1C1 * C1[0]
    # centerLaserB1C1 = [0, bB1C1]
    ApcenterLaserApDp = np.linalg.norm(np.array(Ap) - np.array(centerLaserApDp))
    print(f'A1centerLaserA1D1: {ApcenterLaserApDp}')
    teta = np.arccos(Ap[0]/ApcenterLaserApDp)
    print(f'Teta: {teta}')
    print(f'geometric deviation of the laser beam from the X axis UF1: {np.degrees(teta)} degrees')

    newFig.add_trace(go.Scatter(x=[Ap[0], Dp[0], Bp[0], Cp[0], centerLaserApDp[0], centerLaserBpCp[0]],
                                y=[Ap[1], Dp[1], Bp[1], Cp[1], centerLaserApDp[1], centerLaserBpCp[1]],
                                mode='markers', marker=dict(size=3, color='black', showscale=False),
                                name=f'Ap, Dp, Bp, Cp, cLaserApDp, cLaserBpCp',
                                text=['Ap', 'Dp', 'Bp', 'Cp', 'cLaserApDp', 'cLaserBpCp']))
    newFig.add_trace(go.Scatter(x=[Bp[0], Cp[0]], y=[Bp[1], Cp[1]],
                                mode='lines', line=dict(color='black', width=1), line_dash='dash',
                                name=f'BpCp'))
    newFig.add_trace(go.Scatter(x=[Ap[0], Dp[0]], y=[Ap[1], Dp[1]],
                                mode='lines', line=dict(color='black', width=1), line_dash='dash',
                                name=f'ApDp'))
    newFig.add_trace(go.Scatter(x=[Ap[0], O[0]], y=[Ap[1], O[1]],
                                mode='lines', line=dict(color='red', width=1), line_dash='dash',
                                name=f'OAp'))
    newFig.add_trace(go.Scatter(x=[Dp[0], O[0]], y=[Dp[1], O[1]],
                                mode='lines', line=dict(color='red', width=1), line_dash='dash',
                                name=f'ODp'))
    newFig.add_trace(go.Scatter(x=[centerLaserApDpOld[0]], y=[centerLaserApDpOld[1]],
                                mode='markers', marker=dict(size=3, color='red', showscale=False),
                                name=f'real centerLaserApDp'))


    # 3D points
    scene = dict(
        # camera=dict(eye=dict(x=1.15, y=1.15, z=0.8)),
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
        aspectmode='data'
        # aspectratio=dict(x=1, y=1, z=1)
    )
    layout = go.Layout(scene=scene)
    fig3D = go.Figure(go.Scatter3d(), layout=layout)

    Op.append(0)
    O1 = Op.copy()
    O1[2] = gammaH

    Ap.append(0)
    A1 = Ap.copy()
    Dp.append(0)
    D1 = Dp.copy()

    Bp.append(0)
    B1 = Bp.copy()
    B1[2] = h
    Cp.append(0)
    C1 = Cp.copy()
    C1[2] = h

    centerLaserApDp.append(0)
    centerLaserA1D1 = centerLaserApDp.copy()
    centerLaserBpCp.append(0)
    centerLaserB1C1 = centerLaserBpCp.copy()
    centerLaserB1C1[2] = h

    npGamma1 = matrices.angle2vectors(centerLaserA1D1, centerLaserB1C1, centerLaserA1D1, centerLaserBpCp)
    print(f'npGamma1: {np.degrees(npGamma1)}')
    npGamma2 = matrices.angle2vectors(centerLaserA1D1, O1, centerLaserA1D1, Op)
    print(f'npGamma2: {np.degrees(npGamma2)}')
    norm = matrices.angle2vectors(centerLaserBpCp, centerLaserB1C1, centerLaserBpCp, centerLaserApDp)
    print(f'norm: {np.degrees(norm)}')
    fig3D.add_trace(go.Scatter3d(x=[centerLaserB1C1[0]],
                                 y=[centerLaserB1C1[1]],
                                 z=[centerLaserB1C1[2]],
                                 name='',
                                 mode='markers',
                                 marker=dict(symbol='circle-open', size=3, color='red')
                                 )
                    )


    #main points
    fig3D.add_trace(go.Scatter3d(x=[O1[0], Op[0], centerLaserApDp[0], A1[0], D1[0], B1[0], C1[0], Bp[0], Cp[0]],
                                 y=[O1[1], Op[1], centerLaserApDp[1], A1[1], D1[1], B1[1], C1[1], Bp[1], Cp[1]],
                                 z=[O1[2], Op[2], centerLaserApDp[2], A1[2], D1[2], B1[2], C1[2], Bp[2], Cp[2]],
                                 name='main points',
                                 mode='markers',
                                 text=['O1', 'Op', 'centerLaserAD', 'A1', 'D1', 'B1', 'C1', 'Bp', 'Cp'],
                                 marker=dict(size=2, color='green')
                                 )
                    )
    #vector
    fig3D.add_trace(go.Scatter3d(x=[O1[0], centerLaserA1D1[0]],
                                 y=[O1[1], centerLaserA1D1[1]],
                                 z=[O1[2], centerLaserA1D1[2]],
                                 name='Z axis of the laser',
                                 line=dict(color='red', width=3),
                                 marker=dict(symbol='circle-open', size=3, color='red')
                                 )
                    )

    # fig3D.add_trace(go.Cone(x=[centerLaserAD[0]],
    #                              y=[centerLaserAD[1]],
    #                              z=[centerLaserAD[2]],
    #                              u=[0], v=[0], w=[20]
    #                              )
    #                 )

    # fig3D.add_trace(go.Scatter3d(x=[O[0], centerLaserAD[0]],
    #                              y=[O[1], centerLaserAD[1]],
    #                              z=[O[2], centerLaserAD[2]],
    #                              mode='lines+markers',
    #                              marker=dict(symbol=["arrow-right"], size=3, color='green', angleref="previous",)
    #                              )
    #                 )

    #section
    fig3D.add_trace(go.Mesh3d(x=[B1[0], C1[0], Cp[0], Bp[0]],
                              y=[B1[1], C1[1], Cp[1], Bp[1]],
                              z=[B1[2], C1[2], Cp[2], Bp[2]],
                              i=[0, 0],
                              j=[1, 2],
                              k=[2, 3],
                              opacity=0.5,
                              name='experimental standard cross section'
                              )
                    )

    #laser beam
    fig3D.add_trace(go.Mesh3d(x=[O1[0], Bp[0], Cp[0], B1[0], C1[0], A1[0], D1[0]],
                              y=[O1[1], Bp[1], Cp[1], B1[1], C1[1], A1[1], D1[1]],
                              z=[O1[2], Bp[2], Cp[2], B1[2], C1[2], A1[2], D1[2]],
                              i=[0],
                              j=[5],
                              k=[6],
                              opacity=0.3,
                              color='blue',
                              name='laser beam'
                              )
                    )


    new_row = {"frame": currentFrame[-1].split('\\')[-1],
               "UF1 Ry": data[4][4]/10000,
               "UF1 Rz": data[4][5] / 10000,
               "Teta(Z and UF1)": np.degrees(teta),
               "Gamma(X and UF1)": 90 - np.degrees(gamma),
               "h calculated": np.abs((hCalc2+hCalc1)/2),
               "top edge": BC,
               "errO": errorO,
               "errOx": errorOx,
               "errOy": errorOy,
               "errA": errorA,
               "errD": errorD
               }
    df.loc[len(df)] = new_row
    rowData = df.to_dict("records")

    return f'dX: {round(np.degrees(teta), 4)}, dZ: {round(90 - np.degrees(gamma), 4)}', newFig, fig3D, rowData

#Restore surface
@callback(
    Output('fig3D', 'figure', allow_duplicate=True),
    Input('restSurf', 'n_clicks'),
    State('loadPaths2', 'value'),
    prevent_initial_call=True
)
def restSurf(n_clicks, path):
    j=0
    # 3D points
    scene = dict(
        camera=dict(eye=dict(x=1.15, y=1.15, z=0.8)),
        aspectmode='data'
        # aspectratio=dict(x=1, y=1, z=1)
    )
    layout = go.Layout(scene=scene)
    fig3D = go.Figure(go.Scatter3d(), layout=layout)
    print(f'Current path: {path}')
    frames = [f.path for f in os.scandir(path) if f.is_file() and f.path.split('.')[-1].lower() == 'txt']
    X = np.array([0]*944)
    Y = np.array([0]*944)
    Z = np.array([0]*944)
    for i, line in enumerate(frames):
        arr = []
        with open(line, 'r') as file:
            for line2 in file:
                arr.append([float(x) for x in line2.split()])
        fname = line.split('\\')[-1]
        # print(len(arr[0]))
        X = np.vstack([X, np.array(arr[0])])
        # print(f'X shape: {X.shape}')
        Y = np.vstack([Y, np.full(len(arr[0]),arr[4][0]/1000-450)])
        # print(f'Y shape: {Y.shape}')
        Z = np.vstack([Z, np.array(np.full(len(arr[0]), arr[1]))])
        # print(f'Z shape: {Z.shape}')

    fig3D.add_trace(go.Surface(x=X, y=Y, z=Z))

    return fig3D

#Calculate coordinates
@callback(
    Output('coordinates', 'children'),
    Input('calcCoord', 'n_clicks'),
    State('loadPaths2', 'value'),
    prevent_initial_call=True
)
def calcCoord(n_clicks, path):
    centersHole = []
    print(f'Current path: {path}')
    frames = [f.path for f in os.scandir(path) if f.is_file() and f.path.split('.')[-1].lower() == 'txt']
    for i, line in enumerate(frames):
        arr = []
        with open(line, 'r') as file:
            for line2 in file:
                arr.append([float(x) for x in line2.split()])
        fname = line.split('\\')[-1]
        # print(arr)
        print(f'Calculation for frame: {fname}')
        surf = np.array(arr[0])
        # print(surf.nonzero())
        # print(surf.shape)
        startHoles = []
        [startHoles.append([el, i]) for i, el in enumerate(surf[:-1]) if (el - surf[i + 1]) > 1]
        print(startHoles)
        endHoles = []
        # endHoles.append([surf[el[1]+1], el[1]+1])
        endHoles = [[surf[el[1]+1], el[1]+1] for el in startHoles]
        print(endHoles)
        centersHole.append([(startHoles[0][0]-endHoles[0][0])/2+endHoles[0][0], -(startHoles[1][0]-endHoles[1][0])/2+startHoles[1][0]])

        # res = next(([el, i] for i, el in enumerate(surf[:]) if (el-surf[i+1])>1), -1)
        # if res==-1:
        #     print('nothing')
        # else:
        #     print('ok')
        #     print(res)
        # res = next(([el, i] for i, el in enumerate(surf[:]) if (el - surf[i + 1]) > 1), -1)
        # if res == -1:
        #     print('nothing')
        # else:
        #     print('ok')
        #     print(res)
    print(f'final result: {centersHole}')
    return (0, 0)