#!/usr/bin/env python3
# coding:utf-8
"""
  Author:  LANL Clinic 2019 --<lanl19@cs.hmc.edu>
  Purpose: Attempt to follow a peak.
  Created: 10/18/19
"""

import numpy as np
from spectrogram import Spectrogram
from follower import Follower


class PeakFollower(Follower):
    """

    """

    def __init__(self, spectrogram, start_point, span=30):
        super().__init__(spectrogram, start_point, span)

    def step(self):
        """
        Attempt to fit a gaussian starting from the coeffients
        in the input parameter to intensities vs velocities.
        If coefficients is None or empty, guess reasonable values.
        The order of the parameters in the coefficients array is
        amplitude, center, width, and background.
        """
        velocities, intensities, p_start, p_end = self.data()
        
        # Should we smooth first?
        if True:
            smooth = np.zeros(len(intensities))
            n_window = 3
            for roll in range(-n_window, n_window+1):
                smooth += np.roll(intensities, roll)
            smooth /= 2 * n_window + 1
            intensities = smooth
        
        low_to_high = np.argsort(intensities)
        
        v_high = velocities[low_to_high[-1]]
        
        # We should now determine whether the fit was successful
        if False:
            print("Failed in gaussian fit")
            return False
        else:
            # add this to our results
            self.results['v_spans'].append((p_start, p_end))
            self.results['p_times'].append(self.p_time)
            self.results['times'].append(
                self.spectrogram._point_to_time(self.p_time))
            self.results['velocities'].append(v_high)
            self.p_time += 1
        return True

    def show_fit(self, axes, n=-1):
        """
        Update the axes to show the data and fit corresponding to point n.
        """
        p_start, p_end = self.data_range(n)
        velocities = self.velocity[p_start:p_end]
        intensities = self.intensity[p_start:p_end,
                                     self.results['p_times'][n]]
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
    p_time = spectrogram._time_to_point(t)

    params = []
    try:
        p_velocity = spectrogram._velocity_to_point(center)
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



