#!/usr/bin/env python3
# coding:utf-8
"""
::

  Author:  LANL Clinic 2019 --<lanl19@cs.hmc.edu>
  Purpose: Attempt to follow a peak.
  Created: 10/18/19
"""

import numpy as np
from follower import Follower


class PeakFollower(Follower):
    """
    A naive follower implementation that implements a local-region
    smoothing and then follows the local maximum.

    **Inputs to the constructor**

    - spectrogram: an instance of Spectrogram
    - start_point: (t, v), the coordinates at which to begin the search
    - span: (60) the number of pixels on either side of the starting value of v
      at which to search for a peak at the next time step.
    - smoothing: (4) the number of points on either side of a given velocity
      in a spectrum to average over to produce a smoothed representation of the
      spectrum prior to searching for a peak.
    - max_hop: (50) the largest change in v from the previous time step to consider
      as a continuation.


    """

    def __init__(self, spectrogram, start_point, span=60,
                 smoothing=4,  # average this many points on each side
                 max_hop=50   # require peaks at successive time steps
                 # to be within this many velocity indices
                 ):
        super().__init__(spectrogram, start_point, span)
        self.smoothing = smoothing
        self.max_hop = max_hop

    def run(self):
        """
        Repeatedly call step until it fails.
        """
        while self.step():
            pass

    def step(self):
        """
        Attempt to fit a gaussian starting from the coeffients
        in the input parameter to intensities vs velocities.
        If coefficients is None or empty, guess reasonable values.
        The order of the parameters in the coefficients array is
        amplitude, center, width, and background.
        """
        velocities, intensities, p_start, p_end = self.data()

        if self.smoothing:
            smooth = np.zeros(len(intensities))
            for roll in range(-self.smoothing, self.smoothing + 1):
                smooth += np.roll(intensities, roll)
            smooth /= 2 * self.smoothing + 1
            intensities = smooth

        # generate an index array to sort the intensities (low to high)
        low_to_high = np.argsort(intensities)
        top = low_to_high[-1]  # index of the tallest intensity peak
        v_high = velocities[top]

        # add this to our results
        self.results['velocity_index_spans'].append((p_start, p_end))
        self.results['time_index'].append(self.time_index)
        self.results['times'].append(
            self.spectrogram._point_to_time(self.time_index))
        self.results['velocities'].append(velocities[top])
        self.results['intensities'].append(intensities[top])
        # on success we increment to the next time index
        self.time_index += 1

        if self.time_index >= len(self.time):
            return False  # we're out of data

        try:
            hop = v_high - self.results['velocities'][-2]
            hop = np.abs(hop / (velocities[1] - velocities[0]))
            return hop <= self.max_hop
        except IndexError:
            return True
        except Exception as eeps:
            print(eeps)
            return False


