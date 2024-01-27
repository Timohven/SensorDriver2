import numpy as np

def matrix_equation(X,y):
    a = np.dot(X.T, X)
    b = np.dot(X.T, y)
    return np.linalg.solve(a, b)

def line(x1, y1, x2, y2):
    k = (y1 - y2) / (x1 - x2)
    b = y2 - k * x2
    return k, b