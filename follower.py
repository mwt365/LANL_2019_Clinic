#!/usr/bin/env python3
# coding:utf-8
"""
  Author:  LANL Clinic 2019 --<lanl19@cs.hmc.edu>
  Purpose: Attempt to follow a peak.
  Created: 10/18/19
"""

import numpy as np
from spectrogram import Spectrogram

class Follower:
    """
    Given a spectrogram and a starting point (t, v),
    follow the local maximum through time. Span describes
    the number of points on either side of the current velocity peak to 
    consider in the sample.
    """

    def __init__(self, spectrogram, start_point, span=80):
        assert isinstance(spectrogram, Spectrogram)
        assert isinstance(span, int)
        self.spectrogram = spectrogram
        self.t_start = start_point[0]
        self.v_start = start_point[1]
        self.span = span
        
        # Now establish storage for intermediate results and
        # state. p_time is the index into the spectrogram.intensity
        # along the time axis.
        self.p_time = spectrogram._time_to_point(self.t_start)

        self.results = dict(
            v_spans=[],        # the range of points used for the fit
            times=[],          # the times of the fits
            p_times=[],        # the corresponding point index in the time dimension
            velocities=[],     # the fitted center velocities
        )
        # for convenience, install pointers to useful fields in spectrogram
        self.velocity = self.spectrogram.velocity
        self.time = self.spectrogram.time
        self.intensity = self.spectrogram.intensity
        
        # params stores the 
        # self.params = []

    def data_range(self, n=-1):
        "Fetch the span of velocities and intensities to use for fitting"
        if len(self.results['velocities']) > 0:
            last_v = self.results['velocities'][n]
        else:
            last_v = self.v_start
        p_vel = self.spectrogram._velocity_to_point(last_v)
        p_start = max(0, p_vel - self.span)
        p_end = min(p_vel + self.span, len(self.spectrogram.velocity))
        return (p_start, p_end)
    
    def data(self, n=-1):
        p_start, p_end = self.data_range(n)
        velocities = self.velocity[p_start:p_end]
        intensities = self.intensity[p_start:p_end, self.p_time]
        return velocities, intensities, p_start, p_end





