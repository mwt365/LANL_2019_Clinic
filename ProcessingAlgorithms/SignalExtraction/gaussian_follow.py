#!/usr/bin/env python3
# coding:utf-8
"""
  Author:  LANL Clinic 2019 --<lanl19@cs.hmc.edu>
  Purpose: Attempt to follow a peak.
  Created: 10/18/19

.. module:: gaussian_follow
"""

import numpy as np
from spectrogram import Spectrogram
from scipy.optimize import curve_fit
from ProcessingAlgorithms.SignalExtraction.follower import Follower


def _gauss(x, *p):
    """Gaussian fitting function used by curve_fit"""
    A, mu, sigma, background = p
    return background + A * \
        np.exp(-0.5 * ((x - mu) / sigma)**2)


class GaussianFitter(Follower):
    """

    """

    def __init__(self, spectrogram, start_point, span=80):
        super().__init__(spectrogram, start_point, span)

        self.params = []

    @property
    def coefficients(self):
        return np.array(self.params)

    @coefficients.setter
    def coefficients(self, params):
        assert isinstance(params, (list, tuple))
        self.params = params

    def fit_gaussian(self):
        """
        Attempt to fit a gaussian starting from the coeffients
        in the input parameter to intensities vs velocities.
        If coefficients is None or empty, guess reasonable values.
        The order of the parameters in the coefficients array is
        amplitude, center, width, and background.
        """
        p_start, p_end = self.data_range()
        velocities = self.velocity[p_start:p_end]
        intensities = self.intensity[p_start:p_end, self.time_index]
        # make sure we are dealing with power
        powers = self.spectrogram.power(intensities)

        # Should we smooth first?
        # if True:

        if len(self.coefficients) != 4:
            avg = np.mean(powers)
            amp = np.max(powers) - avg
            # if no center was specified, use the middle
            # of the range
            center = velocities[len(velocities) // 2]
            width = velocities[10] - velocities[0]  # this is strange
            self.coefficients = [amp, center, width, avg]

        coeff, var_matrix = curve_fit(
            _gauss, velocities, powers,
            p0=self.coefficients
        )
        # We should now determine whether the fit was successful
        if np.nan in var_matrix:
            print("Failed in gaussian fit")
            return False
        else:
            # add this to our results
            self.results['time_index'].append(self.time_index)
            self.results['times'].append(
                self.spectrogram._point_to_time(self.time_index))
            self.results['velocities'].append(coeff[1])
            self.results['widths'].append(coeff[2])
            self.results['amplitudes'].append(coeff[0])
            self.results['backgrounds'].append(coeff[3])
        return True

    def show_fit(self, axes, n=-1):
        """
        Update the axes to show the data and fit corresponding to point n.
        """
        p_start, p_end = self.data_range(n)
        velocities = self.velocity[p_start:p_end]
        intensities = self.intensity[p_start:p_end,
                                     self.results['time_index'][n]]
        powers = self.spectrogram.power(intensities)

        axes.lines = []
        # plot the data
        axes.plot(velocities, powers, 'ro-', alpha=0.4)
        # compute the fitted curve
        # A, mu, sigma, background = p
        params = [self.results[x][n] for x in
                  ('amplitudes', 'velocities', 'widths', 'backgrounds')]
        curve = _gauss(velocities, *params)
        axes.plot(velocities, curve, 'b-', alpha=0.8)


def gaussian_follow(spectrogram, start):
    """
    Pass in a spectrogram and a starting point start=(t, v)
    """
    assert isinstance(spectrogram, Spectrogram)
    t, center = start  # the time and the rough peak position
    intensity = spectrogram.intensity
    velocity = spectrogram.velocity

    # map the time and velocity to a point
    p_time = spectrogram._time_to_index(t)

    params = []
    try:
        p_velocity = spectrogram._velocity_to_index(center)
        start, end = p_velocity - span, p_velocity + span
        params = fit_gaussian(
            velocity[start:end],
            intensity[start:end, p_time],
            params
        )
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



