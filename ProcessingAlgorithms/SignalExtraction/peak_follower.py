#!/usr/bin/env python3
# coding:utf-8
"""
::

  Author:  LANL Clinic 2019 --<lanl19@cs.hmc.edu>
  Purpose: Attempt to follow a peak.
  Created: 10/18/19
"""

import numpy as np
from spectrogram import Spectrogram
from ProcessingAlgorithms.SignalExtraction.follower import Follower
from baselines import baselines_by_squash as bline
from ProcessingAlgorithms.Fitting.moving_average import moving_average


class PeakFollower(Follower):
    """
    A naive follower implementation that uses a local-region
    smoothing and then follows the local maximum.

    **Inputs to the constructor**

    - spectrogram: an instance of Spectrogram
    - start_point: (t, v), the coordinates at which to begin the search
    - span: (80) the number of pixels on either side of the starting value of v
      at which to search for a peak at the next time step.
    - smoothing: (4) the number of points on either side of a given velocity
      in a spectrum to average over to produce a smoothed representation of the
      spectrum prior to searching for a peak.
    - max_hop: (70) the largest change in v from the previous time step to consider
      as a continuation.
    """

    def __init__(self,
                 spectrogram: Spectrogram,
                 start_point,  # either a tuple or a list of tuples
                 span=80,      # width of neighborhood to search
                 smoothing=4,  # average this many points on each side
                 max_hop=500,  # require peaks at successive time steps
                               # to be within this many m/s
                 direction=1,  # go forward in time (1), backward (-1)
                 ):
        super().__init__(spectrogram, start_point, span)
        self.smoothing = smoothing
        self.max_hop = max_hop
        assert direction in (-1, 1), "Direction must be +1 or -1"
        self.direction = direction
        peaks, dv, heights = bline(spectrogram)
        # Let's take only the peaks greater than 10% of the strongest peak
        self.baselines = peaks[heights > 0.1]

    def reverse(self):
        """Switch directions"""
        self.direction = -self.direction
        # now update the value of time_index
        tindex = self.results['time_index']
        if len(tindex) > 0:
            if self.direction > 0:
                self.time_index = tindex[-1] + 1
                self.last_peak = self.results['velocities'][-1]
            else:
                self.time_index = tindex[0] - 1
                self.last_peak = self.results['velocities'][0]

    def run(self):
        """
        Repeatedly call step until it fails.
        """
        failures = 0
        consecutive_failures = 0
        while True:
            try:
                self.step()
                consecutive_failures = 0
                continue
            except IndexError:
                break
            except Exception as eeps:
                failures += 1
                consecutive_failures += 1

            if consecutive_failures > 10:
                break

    def step(self):
        """
        Identify the tallest peak in the range of data supplied
        by the inherited data() method, after removing peaks that
        are too close to known baselines.

        in the input parameter to intensities vs velocities.
        If coefficients is None or empty, guess reasonable values.
        The order of the parameters in the coefficients array is
        amplitude, center, width, and background.
        """
        velocities, intensities, p_start, p_end = self.data(self.time_index)

        if self.smoothing:
            intensities = moving_average(intensities, n=self.smoothing)

        # generate an index array to sort the intensities (low to high)
        low_to_high = np.argsort(intensities)
        high_to_low = np.flip(low_to_high)

        # remove any peaks that are too close to the baseline
        # Since argsort only sorts from low to high, we will use negative
        # indexing to crawl down from the greatest intensity.
        n = 0
        while True:
            vpeak = velocities[high_to_low[n]]
            mingap = np.min(np.abs(self.baselines - vpeak))
            if mingap > 100:
                break
            n += 1
        top = high_to_low[n]
        v_high = velocities[top]

        # At this point, we need a criterion for accepting or rejecting
        # this point.
        reject = False
        if len(self.results['times']) > 0:
            i_guess = self.guess_intensity(self.time_index)
            reject = 0.05 * i_guess > intensities[top]

        hop = None if not hasattr(self, 'last_peak') else abs(
            v_high - self.last_peak)
        if hop and hop > self.max_hop:
            reject = True
        # add this to our results
        if not reject:
            self.add_point(self.time_index, v_high, span = (p_start, p_end))
            try:
                hop = abs(v_high - self.last_peak)
                if hop > self.max_hop:
                    print(f"hop at [{self.time_index}] of {hop} from {self.last_peak:0f} to {self.v_high:0f}")

            except:
                pass
            self.last_peak = v_high

        # on success we increment to the next time index
        self.time_index += self.direction

        if self.time_index >= len(self.time) or self.time_index < 0:
            raise IndexError  # we're out of data



