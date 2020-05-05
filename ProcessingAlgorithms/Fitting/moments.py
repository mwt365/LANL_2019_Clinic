#!/usr/bin/env python3
# coding:utf-8
"""
  Author:  LANL Clinic 2019 --<lanl19@cs.hmc.edu>
  Purpose: Attempt to extract the width of a peak without
           assuming a gaussian shape
  Created: 03/02/20
"""

import numpy as np


def moment(x, y, noise_level=None):
    """
    Given arrays x and y, with x having equally spaced points,
    represent y as a probability distribution and determine
    therefrom the mean and variance.

    We assume that there is a peak, but also that y represents
    intensity values (and is intrinsically nonnegative), but
    there is noise in the surrounding region. To remove its
    influence, sort the y values and average the lower half
    to estimate the zero level.

    If no noise_level is passed in, it is estimated from
    the average of the lower half of the intensities.
    """

    if noise_level == None:
        ysort = np.sort(y)
        bottom = ysort[:(len(ysort) // 2)]
        noise_level = bottom.mean()

    # Construct a normalized quasi-probability distribution
    prob = y - noise_level
    total = np.sum(prob)
    prob /= total
    dx = x[1] - x[0]

    # Now compute the first moment
    x_center = np.dot(x, prob)
    delta_x = x - x_center
    variance = np.dot(delta_x * delta_x, prob)
    std_dev = np.nan if variance < 0.0 else np.sqrt(variance)
    avg_dev = np.dot(np.abs(delta_x), prob)
    npnts = 8 * std_dev / dx
    return dict(
        center=x_center,
        variance=variance,
        std_dev=std_dev,
        avg_dev=avg_dev,
        std_err=std_dev / np.sqrt(npnts),
        background=noise_level
    )
