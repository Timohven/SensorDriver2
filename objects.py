from utilities import matrices
import numpy as np

class Frame2D:
    def __init__(self, X, Z):
        self.O = [0.0, 0.0]
        N = 3  # number of sequential values
        DELTA = 10  # hight difference value
        limY1 = sum(Z[0:N]) / N - DELTA
        limY2 = sum(Z[-N:]) / N - DELTA
        [first, i] = next([el, i] for i, el in enumerate(Z[:]) if el < limY1)
        self.B = [X[i], Z[i]]
        [last, j] = next([el, i] for i, el in enumerate(Z[::-1]) if el < limY2)
        self.C = [X[-j - 1], Z[-j - 1]]
        self.A = [X[i - 1], Z[i - 1]]
        self.D = [X[-j], Z[-j]]

        OA = np.linalg.norm(np.array(self.A) - np.array(self.O))
        ang = np.arctan(self.B[1]/self.B[0])
        print(f'ang1: {ang}')
        self.calcA = [np.cos(ang) * OA, np.sin(ang) * OA]
        self.errorA = np.linalg.norm(np.array(self.A) - np.array(self.calcA))/OA*100

        OD = np.linalg.norm(np.array(self.D) - np.array(self.O))
        ang = np.abs(np.arctan(self.C[1] / self.C[0]))
        print(f'ang2: {ang}')
        self.calcD = [-(np.cos(ang) * OD), np.sin(ang) * OD]
        self.errorD = np.linalg.norm(np.array(self.D) - np.array(self.calcD))/OD*100

        print(f'calcA: {self.calcA}, calcD: {self.calcD}')
        # kOB, bOB = matrices.line(self.O, self.B)
        # kOC, bOC = matrices.line(self.O, self.C)



    def __str__(self):
        return f'A: {self.A}, B: {self.B}, C: {self.C}, D: {self.D}, errorA: {self.errorA}, errorD: {self.errorD}, '
