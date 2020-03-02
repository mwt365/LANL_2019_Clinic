#!/usr/bin/env python3
# coding:utf-8
"""
  Author:  LANL Clinic 2019 --<lanl19@cs.hmc.edu>
  Purpose: Attempt to follow a peak.
  Created: 10/18/19
"""

import numpy as np
import pandas as pd
from spectrogram import Spectrogram


class Follower:
    """
    Given a spectrogram and a starting point (t, v), a follower
    looks for a quasicontinuous local maximum through the spectrogram.
    This base class handles storage of the spectrogram reference, the
    starting point, and a span value describing the width of the
    neighborhood to search, centered on the previous found maximum.

    It also holds a results dictionary with several obligatory
    fields, to which a subclass may add. The required fields are

        velocity_index_spans: the range of point indices used
        times:                the times found (s)
        time_index:           the index of the time columns
        velocities:           the peak velocities
        intensities:          the intensity at the peak


    """

    def __init__(self, spectrogram, start_point, span=80):
        self.spectrogram = spectrogram
        self.t_start = start_point[0]
        self.v_start = start_point[1]
        self.span = span

        # Now establish storage for intermediate results and
        # state. time_index is the index into the spectrogram.intensity
        # along the time axis.
        self.time_index = spectrogram._time_to_index(self.t_start)

        self.results = dict(
            velocity_index_spans=[],  # the range of points used for the fit
            times=[],                 # the times of the fits
            time_index=[],            # the corresponding point index in the time dimension
            velocities=[],            # the fitted center velocities
            intensities=[]            # the peak intensity
        )
        # for convenience, install pointers to useful fields in spectrogram
        self.velocity = self.spectrogram.velocity
        self.time = self.spectrogram.time
        self.intensity = self.spectrogram.intensity

    @property
    def v_of_t(self):
        "A convenience function for plotting; returns arrays for time and velocity"
        t = np.array(self.results['times'])
        v = np.array(self.results['velocities'])
        return t, v

    def data_range(self, n=-1):
        """
        Fetch the span of velocities and intensities to use for fitting.
        The default value of n (-1) indicates the last available point
        from the results dictionary. Earlier points are possible.
        """
        if len(self.results['velocities']) > 0:
            last_v = self.results['velocities'][n]
        else:
            last_v = self.v_start
        velocity_index = self.spectrogram._velocity_to_index(last_v)
        start_index = max(0, velocity_index - self.span)
        end_index = min(velocity_index + self.span,
                        len(self.spectrogram.velocity))
        return (start_index, end_index)

    def data(self, n=-1):
        start_index, end_index = self.data_range(n)
        velocities = self.velocity[start_index:end_index]
        intensities = self.intensity[start_index:end_index, self.time_index]
        return velocities, intensities, start_index, end_index

    @property
    def frame(self):
        """
        Return a pandas DataFrame holding the results of this
        following expedition, with an index of the times
        converted to microseconds.
        """
        microseconds = np.array(self.results['times']) * 1e6
        return pd.DataFrame(self.results, index=microseconds)



