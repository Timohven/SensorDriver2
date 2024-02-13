from utilities import matrices
import numpy as np

class Frame2D:
    def __init__(self, X, Z):
        self.O = [0.0, 0.0]
        N = 3  # number of sequential values
        DELTA = 10  # height difference value
        limY1 = sum(Z[0:N]) / N - DELTA
        limY2 = sum(Z[-N:]) / N - DELTA
        [first, i] = next([el, i] for i, el in enumerate(Z[:]) if el < limY1)
        self.B = [X[i], Z[i]]
        [last, j] = next([el, j] for j, el in enumerate(Z[::-1]) if el < limY2)
        self.C = [X[-j - 1], Z[-j - 1]]
        self.A = [X[i - 1], Z[i - 1]]
        self.D = [X[-j], Z[-j]]

        self.indexA = i-1
        self.indexB = i
        self.indexC = -j-1
        self.indexD = -j

        self.calcA = [0.0, 0.0]
        self.calcD = [0.0, 0.0]

        self.approxB = [0.0, 0.0]
        self.approxC = [0.0, 0.0]

        self.approxA = [0.0, 0.0]
        self.approxD = [0.0, 0.0]

    def __str__(self):
        return f'A: {self.A}, B: {self.B}, C: {self.C}, D: {self.D}, errorA: {self.errorA}, errorD: {self.errorD}, '

    #calculation A and D using angle
    # def calcShadow(self):
    #     OA = np.linalg.norm(np.array(self.A) - np.array(self.O))
    #     ang = np.arctan(self.approxB[1] / self.approxB[0])
    #     # print(f'ang1: {ang}')
    #     self.calcA = [np.cos(ang) * OA, np.sin(ang) * OA]
    #     # self.errorA = np.linalg.norm(np.array(self.A) - np.array(self.calcA)) / OA * 100
    #     self.errorA = np.linalg.norm(np.array(self.A) - np.array(self.calcA))
    #
    #     OD = np.linalg.norm(np.array(self.D) - np.array(self.O))
    #     ang = np.abs(np.arctan(self.approxC[1] / self.approxC[0]))
    #     # print(f'ang2: {ang}')
    #     self.calcD = [-(np.cos(ang) * OD), np.sin(ang) * OD]
    #     # self.errorD = np.linalg.norm(np.array(self.D) - np.array(self.calcD)) / OD * 100
    #     self.errorD = np.linalg.norm(np.array(self.D) - np.array(self.calcD))

    #approximation of a shadow (edge AD) using a straight line

    def shadowApproximation(self, X, Z):
        _A = slice(None, self.indexA)
        D_ = slice(self.indexD, None)
        allX = X[_A] + X[D_]
        x = np.array(allX)
        allZ = Z[_A] + Z[D_]
        y = np.array(allZ)
        a, b = matrices.linearApproximation(x, y)
        self.approxA = [X[self.indexA], a + X[self.indexA] * b]
        self.approxD = [X[self.indexD], a + X[self.indexD] * b]
        self.errorA = np.linalg.norm(np.array(self.A) - np.array(self.approxA))
        self.errorD = np.linalg.norm(np.array(self.D) - np.array(self.approxD))

        return a, b

    # finding the point of intersection of lines AB and DC
    # a1x+b1y=c1; a2x+b2y=c2
    # x-y=0; -x-y=-1
    # -kx+y=b
    # k1, b1 = matrices.line(*A, *B)
    # k2, b2 = matrices.line(*D, *C)
    # mM = np.matrix([[-k1, 1], [-k2, 1]])
    # mC = np.matrix([b1, b2])
    # mO = mM.I.dot(mC.T)
    # O = np.asarray(mO).reshape(-1).tolist()

    # calculation A and D using intersection of lines
    def calcShadow(self):
        # finding the point of intersection of lines AB and DC
        # a1x+b1y=c1; a2x+b2y=c2
        # x-y=0; -x-y=-1
        # -kx+y=b
        self.calcA = matrices.linesIntersection(self.approxA, self.approxD, self.O, self.approxB)
        self.calcD = matrices.linesIntersection(self.approxA, self.approxD, self.O, self.approxC)

        self.errorA = np.linalg.norm(np.array(self.calcA) - np.array(self.approxA))
        if self.calcA[0] < self.approxA[0]: self.errorA = - self.errorA
        self.errorD = np.linalg.norm(np.array(self.calcD) - np.array(self.approxD))
        if self.calcD[0] < self.approxD[0]: self.errorD = - self.errorD

    #approximation of a face (edge BC) using a straight line
    def faceApproximation(self, X, Z):
        x = np.array(X[self.indexB: self.indexC])
        y = np.array(Z[self.indexB: self.indexC])
        a, b = matrices.linearApproximation(x, y)
        self.approxB = [X[self.indexB], a + X[self.indexB]*b]
        self.approxC = [X[self.indexC], a + X[self.indexC]*b]

        return a, b
