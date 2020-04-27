# coding:utf-8

"""
::

   Author:  LANL Clinic 2019 --<lanl19@cs.hmc.edu>
   Purpose: Simple smoothing routine
   Created: 11/17/19
"""
import numpy as np


def moving_average(x: np.ndarray, n=3, shape='triangle'):
    """
    Prepare a smoothed version by averaging values
    from n to the left to n to the right. Shape may be one
    of {rectangle, triangle, gaussian}
    """
    pts = 2 * n + 1
    offsets = np.linspace(-n, n, pts, dtype=np.int32)
    if shape == 'triangle':
        weights = -np.abs(offsets) + n + 1
    elif shape == 'gaussian':
        weights = np.exp(-0.5 * np.linspace(-1.25, 1.25, pts)**2)
    else:
        weights = np.ones(pts)
    total = np.sum(weights)
    weights = np.asarray(weights, dtype=np.float) / total

    sm = np.zeros(len(x))
    for i, w in zip(offsets, weights):
        sm += w * np.roll(x, i)
    return sm


def compress(num_pnts: int, *args):
    """
    Produce a compressed representation of the np.ndarrays in args
    by averaging num_pnts consecutive values. 
    """
    res = []
    for arg in args:
        assert isinstance(arg, np.ndarray)
        chunks = len(arg) // num_pnts  # how many points we'll get
        pts = chunks * num_pnts
        resh = np.reshape(arg[0:pts], (chunks, num_pnts))
        x = np.mean(resh, axis=1)
        res.append(x)
    return res

def normalize(b):
    """
    Produce a verions of the matrix b that has 0 mean and 1 sample standard deviation.
    """
    avg = np.mean(b)
    std = np.std(b, ddof= 1)
    return (b-avg)/std


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from numpy.random import Generator, PCG64
    rg = Generator(PCG64())
    x = np.linspace(-4, 4, 161)
    noisy = rg.standard_normal(len(x))
    noisy += 8 * np.exp(-1.0 * (x + 1)**2)
    noisy += 20 * np.exp(-8.0 * (x - 1)**2)
    plt.plot(x, noisy, 'ko', alpha=0.5)
    shapes = ('rectangle', 'triangle', 'gaussian')
    for shape in shapes:
        plt.plot(x, moving_average(noisy, n=10, shape=shape))
    plt.legend(['data'] + list(shapes))
    plt.show()
