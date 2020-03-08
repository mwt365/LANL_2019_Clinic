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
    - max_acceleration: (2000 m/s/µs), used to determine how many pixels
      on either side of the starting value of v
    - smoothing: (4) the number of points on either side of a given velocity
      in a spectrum to average over to produce a smoothed representation of the
      spectrum prior to searching for a peak.
    - max_hop: (70) the largest change in v from the previous time step to consider
      as a continuation.
    - min_intensity_drop: (0.01)
    """

    def __init__(self,
                 spectrogram: Spectrogram,
                 start_point,  # either a tuple or a list of tuples
                 max_acceleration=1000,
                 smoothing=4,  # average this many points on each side
                 max_hop=500,  # require peaks at successive time steps
                               # to be within this many m/s
                 direction=1,  # go forward in time (1), backward (-1)
                 ):
        span = max_acceleration * spectrogram.dt * 1e6 / spectrogram.dv
        super().__init__(spectrogram, start_point, int(span))
        self.smoothing = smoothing
        self.max_hop = max_hop
        assert direction in (-1, 1), "Direction must be +1 or -1"
        self.direction = direction
        peaks, dv, heights = bline(spectrogram)
        # Let's take only the peaks greater than 10% of the strongest peak
        self.baselines = peaks[heights > 0.1]
        self.noise_level = spectrogram.noise_level() * 3

    def reverse(self):
        """Switch directions"""
        self.direction = -self.direction
        # now update the value of time_index
        tindex = self.results['time_index']
        if len(tindex) > 0:
            if self.direction > 0:
                self.time_index = tindex[-1] + 1
                self.last_peak = self.results['peak_velocity'][-1]
            else:
                self.time_index = tindex[0] - 1
                self.last_peak = self.results['peak_velocity'][0]

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
        USE_INTENSITIES = False
        velocities, intensities, p_start, p_end = self.data(self.time_index)

        if self.smoothing:
            intensities = moving_average(intensities, n=self.smoothing)

        # generate an index array to sort the intensities (low to high)
        high_to_low = np.flip(np.argsort(intensities))

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
        if len(self.results['time']) > 0:
            reject = intensities[top] < self.noise_level
        if not reject and USE_INTENSITIES:
            i_guess = self.guess_intensity(self.time_index)
            reject = 0.01 * i_guess > intensities[top]
            if reject:
                i_guess = self.guess_intensity(self.time_index)

        hop = None if not hasattr(self, 'last_peak') else abs(
            v_high - self.last_peak)
        if hop and hop > self.max_hop:
            reject = True

        if not reject:
            self.add_point(self.time_index, v_high, span=(p_start, p_end))
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
        if reject:
            raise Exception("failed")


def signal_finder(
        spectrogram: Spectrogram,
        baseline: np.ndarray,
        t_start: float,
        dt: int = 2,
        min_separation=200):  # m/s
    """
    Look above the baseline for a signal corresponding to a surface velocity.
    """
    from scipy.signal import find_peaks

    t_index = spectrogram._time_to_index(t_start)
    spectra = np.sum(spectrogram.intensity[:, t_index - dt:t_index + dt],
                     axis=1)
    # limit the region we look at to points above the baseline
    # get the index of the baseline
    blo = spectrogram._velocity_to_index(baseline[0]) + 5
    if len(baseline) > 1:
        bhi = spectrogram._velocity_to_index(baseline[1]) - 5
    else:
        bhi = len(spectrogram.velocity) - 1
    velocity = spectrogram.velocity[blo:bhi]
    spectrum = spectra[blo:bhi]
    smax = spectrum.max()
    min_sep = int(min_separation / spectrogram.dv)
    peaks, properties = find_peaks(spectrum, height=0.05 * smax,
                                   distance=min_sep)

    try:
        heights = properties['peak_heights']
        # produce an ordering of the peaks from high to low
        ordering = np.flip(np.argsort(heights))
        peak_index = peaks[ordering]
        peaks = velocity[peak_index]
        hts = heights[ordering]

        # normalize to largest peak of 1 (e.g., the baseline)
        hts = hts / hts[0]

        return peaks[0]
    except Exception as eeps:
        print(eeps)
        return None


