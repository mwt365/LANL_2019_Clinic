#!/usr/bin/env python3
# coding:utf-8
"""
  Author:  LANL Clinic 2019 --<lanl19@cs.hmc.edu>
  Purpose: Attempt to extract the width of a peak without
           assuming a gaussian shape
  Created: 03/02/20
"""

import numpy as np


def moment(x, y):
    """
    Given arrays x and y, with x having equally spaced points,
    represent y as a probability distribution and determine
    therefrom the mean and variance.

    We assume that there is a peak, but also that y represents
    intensity values (and is intrinsically nonnegative), but
    there is noise in the surrounding region. To remove its
    influence, sort the y values and average the lower half
    to estimate the zero level.
    """

    ysort = np.sort(y)
    noise_level = np.mean(ysort[:len(ysort) // 2])
    prob = y - noise_level
    total = np.sum(prob)
    prob /= total

    # Now compute the first moment
    x_center = np.dot(x, prob)
    squares = np.dot(x * x, prob)
    variance = squares - x_center ** 2
    std_dev = np.sqrt(variance)
    return dict(center = x_center, variance = variance, std_dev = std_dev)
