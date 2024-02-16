import numpy as np

def matrix_equation(X,y):
    #x=[B[0], C[0]], y=[a+B[0]*b, a+C[0]*b]
    a = np.dot(X.T, X)
    b = np.dot(X.T, y)
    return np.linalg.solve(a, b)

def line(x1, y1, x2, y2):
    # y = b + k*x

    k = (y1 - y2) / (x1 - x2)
    # if (x1 - x2): k = 0
    # else: k = (y1 - y2) / (x1 - x2)
    b = y2 - k * x2
    return k, b

def linearApproximation(x, y):
    # y = a + b*x
    x_np = x.reshape(-1, 1)
    vector_1 = np.ones((x_np.shape[0], 1))
    x_np = np.hstack((vector_1, x_np))
    y_np = y.reshape(-1, 1)
    ab_np = matrix_equation(x_np, y_np)
    ab_np = np.around(ab_np, 4)
    a = ab_np[0][0]
    b = ab_np[1][0]

    return a, b

def linesIntersection(A, B, C, D):
    # finding the point of intersection of lines AB and DC
    # a1x+b1y=c1; a2x+b2y=c2
    # x-y=0; -x-y=-1
    # -kx+y=b
    k1, b1 = line(*A, *B)
    k2, b2 = line(*C, *D)
    mM = np.matrix([[-k1, 1], [-k2, 1]])
    mC = np.matrix([b1, b2])
    mO = mM.I.dot(mC.T)
    O = np.asarray(mO).reshape(-1).tolist()

    return O

def calcOffsetTF2(h):
    #z,x,Ry from ToolFrame1 in mm and degree
    a = 446.544
    b = 20.127
    alfaDegree = (90-48.5)
    alfa = np.deg2rad(alfaDegree)
    c = b*np.tan(alfa)
    l1 = np.sqrt(c**2 + b**2)
    l2 = h - l1
    print(f'l1, l2: {l1, l2}')
    deltaX = l2 * np.cos(alfa)
    e = l2 * np.sin(alfa)
    print(f'a, c, e: {a, c, e}')
    deltaZ = a + c + e
    return deltaZ, deltaX

def angle2vectors(A, B, C, D):
    a = np.array(A) - np.array(B)
    b = np.array(C) - np.array(D)
    normA = np.linalg.norm(a)
    normB = np.linalg.norm(b)
    print(f'{a, normA, b, normB}')
    angle = np.arccos(a.dot(b)/(normA*normB))
    return angle

def calcOffsetTF2New(h):
    #z,x,Ry from ToolFrame1 in mm and degree
    deltaHTCP = 7.545 #difference between 0 of UF1 and TCP(TF1)
    h += deltaHTCP
    a = 446.544
    b = 20.127
    RyDegree = -48.5
    Ry = np.deg2rad(np.abs(RyDegree))
    l2 = h - b / np.sin(Ry)
    a2 = l2 * np.cos(Ry)
    a0 = b / np.tan(Ry)
    a1 = a + a0 + a2
    b1 = l2 * np.sin(Ry)

    return a1, b1
