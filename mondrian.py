"""
This is a simple implementation of the Mondian kernel, which uses Mondrian processes
to approximate the Laplacian kernel. 
Based on following paper: https://arxiv.org/pdf/1606.05241.pdf

Note that I haven't tested this extensively, so the computations may not be correct.
Pay attention especially to the computation of the class of the point,
which uses a generating function approach. 
"""
from typing import Union

import numpy as np
from numpy.random import exponential, uniform
import matplotlib.pyplot as plt


class Mondrian:
    X: np.ndarray

    def __init__(self, X):
        self.X = X

    def _cut_space(self, d) -> (int, float):
        l, h = self.X[d]
        cut = uniform(low=l, high=h, size=1)[0]
        return d, cut

    def fit(self, lmd: float = 1):
        t = 0
        d = 0
        while d < self.X.shape[0]:
            # always cut at time 0
            t += exponential(scale=lmd, size=1)[0]
            for i in range(2 ** d):
                tupl = self._cut_space(d)
                yield tupl[0], tupl[1], t
            d += 1

    def classify(self, x: np.ndarray, lmd: float = 1) -> Union[np.ndarray, int]:

        if x.ndim > 1:
            d = np.zeros((1, x.shape[0]))[0]
            # print(d)
            for i in range(x.shape[0]):
                d[i] = self._classify_point(x[i, :], lmd)
            return d
        else:
            return self._classify_point(x, lmd)

    def _classify_point(self, x, lmd: float = 1) -> int:
        """
        Classify an input point. It compares the coordinates of the point with the given
        separation lines and constructs a bit sequence (yes-no) from which the class is computed.
        :param x: input point
        :param lmd: lifetime of the mondrian process
        :return:
        """
        cuts = np.array([i[1] for i in self.fit(lmd=lmd)])

        # find the position based on the cuts
        pos = np.zeros((1, x.shape[0]), dtype=np.bool)[0]
        i = 0
        c = 0
        while i < x.shape[0]:
            pos[c] = x[c] > cuts[i]
            i = 2 * i + 1 + pos[c]
            c += 1

        # compute the generating function at the point
        return pos @ (2 ** np.array(range(x.shape[0])))

    def plot_mondrian(self, lmd):
        m = list(self.fit(lmd))

        plt.axvline(x=m[0][1], c='r', ymin=X[1, 0], ymax=X[1, 1])
        plt.hlines(y=m[1][1], xmin=X[0, 0], xmax=m[0][1])
        plt.hlines(y=m[2][1], xmin=m[0][1], xmax=X[0, 1])
        plt.show()


if __name__ == '__main__':
    X = np.array([[1, 3],
                  [2, 4]])

    data = np.array([[1.5, 2.5],
                     [2, 3],
                     [1, 2],
                     [3, 4]])

    d = data.shape[0]

    # number of processes
    m = 10

    # define a signature matrix, holding the class of each point
    signature = np.zeros((m, d))

    for i in range(m):
        mondr = Mondrian(X)
        clss = mondr.classify(data)
        signature[i, :] = clss

    # finally, estimate the kernel
    k = np.zeros((data.shape[0], d))

    # diagonal entries should be 0, no matter what the estimate says

    for i in range(d):
        for j in range(i):
            val = np.sum(signature[i] == signature[j]) / np.sqrt(m)
            k[i, j] = val
            k[j, i] = k[i, j]

    print(k)
