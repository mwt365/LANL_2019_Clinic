#!/usr/bin/env python3
# coding:utf-8
"""
  Author:  LANL Clinic 2019 --<lanl19@cs.hmc.edu>
  Purpose: Attempt to follow a peak.
  Created: 10/18/19
"""

import numpy as np
from spectrogram import Spectrogram
from scipy.optimize import curve_fit


def _gauss(x, *p):
    A, mu, sigma, background = p
    return background + A * \
        np.exp(-0.5 * ((x - mu) / sigma)**2)


def gaussian_follow(spectrogram, start):
    """
    Pass in a spectrogram and a starting point (t, v)
    """
    assert isinstance(spectrogram, Spectrogram)
    t, center = start  # the time and the rough peak position
    intensity = spectrogram.intensity
    velocity = spectrogram.velocity

    # map the time and velocity to a point
    p_time = spectrogram._time_to_point(t)

    def fit_gaussian(velocities, intensities, center=None):

        avg = np.mean(intensities)
        amp = np.max(intensities) - avg
        # if no center was specified, use the middle
        # of the range
        if center == None:
            center = velocities[len(velocities) // 2]
        width = velocities[3] - velocities[0]
        coeff, var_matrix = curve_fit(
            _gauss, velocities, intensities,
            p0=(amp, center, width, avg)
        )
        return coeff

    coefficients = []

    try:
        while True:
            span = 80
            p_velocity = spectrogram._velocity_to_point(center)
            start, end = p_velocity - span, p_velocity + span
            params = fit_gaussian(
                velocity[start:end],
                intensity[start:end, p_time],
                center
            )
            center = params[1]
            values = list(params)
            values.insert(0, spectrogram.time[p_time])
            coefficients.append(values)
            p_time += 1
    except Exception as eeps:
        print(eeps)
    coef = np.array(coefficients)
    return dict(
        time=coef[:, 0] * 1e6,  # converted to microseconds
        amplitude=coef[:, 1],
        center=coef[:, 2],
        width=coef[:, 3],
        background=coef[:, 4]
    )




