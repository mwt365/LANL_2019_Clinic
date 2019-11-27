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
