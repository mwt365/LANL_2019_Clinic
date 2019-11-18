# coding:utf-8

"""
::

   Author:  LANL Clinic 2019 --<lanl19@cs.hmc.edu>
   Purpose: Simple smoothing routine
   Created: 11/17/19
"""
import numpy as np


def moving_average(x: np.ndarray, n=3):
    """
    prepare a smoothed version by averaging values
    from n to the left to n to the right
    """
    sm = np.zeros(len(x))
    for i in range(-n, n + 1):
        sm += np.roll(x, i)
    return sm / (1 + 2 * n)
