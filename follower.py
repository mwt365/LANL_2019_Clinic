# coding:utf-8

"""
::

  Author:  LANL Clinic 2019 --<lanl19@cs.hmc.edu>
  Purpose: Attempt to follow a peak.
  Created: 10/18/19
"""

import numpy as np
import pandas as pd
from spectrogram import Spectrogram
from scipy.optimize import curve_fit


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
        assert isinstance(spectrogram, Spectrogram)
        assert isinstance(span, int)
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

    def estimate_widths(self):
        """
        Assuming that some mechanism has been used to populate
        self.results with a path through the minefield, attempt to
        determine gaussian widths centered on these peaks, or very
        near them. There are a number of games we could play.
        We could alter the number of points used in computing the
        spectra; we could use finer time steps; we could establish
        a noise cutoff level...  In any case, I think it is likely
        most important to use true intensity values, not a logarithmic
        variant.
        """
        res = self.results
        # Prepare a spot for the uncertainties
        res['velocity_uncertainty'] = np.zeros(len(res['velocities']))
        for n in range(len(res['time_index'])):
            fit_res = self.estimate_width(n)
            res['velocities'][n] = fit_res['center']
            res['velocity_uncertainty'][n] = fit_res['width']
            res['intensities'][n] = fit_res['amplitude']

    def estimate_width(self, n, neighborhood=30):
        """
        Estimate the gaussian width of a peak
        """
        res = self.results
        time_index = res['time_index'][n]
        v_peak = res['velocities'][n]
        v_peak_index = self.spectrogram._velocity_to_index(v_peak)
        n_low = max(0, v_peak_index - neighborhood)
        n_high = min(v_peak_index + neighborhood + 1, len(self.velocity))
        # fetch the true power values for this column of the spectrogram
        power = self.spectrogram.power(
            self.intensity[n_low:n_high, time_index])
        res = self.fit_gaussian(
            self.velocity[n_low:n_high], power, v_peak)
        res['power'] = power
        res['indices'] = (n_low, n_high)
        return res

    def fit_gaussian(self, velocities, powers, center):
        """
        Given an array of intensities and a rough location of the peak,
        attempt to fit a gaussian and return the fitting parameters. The
        independent variable is "index" or "pixel" number. We assume that
        the noise level is zero, so we first fit to a gaussian with a
        baseline of 0 and a center of the given location.
        """

        def just_amp_width(x, *p):
            "p = (amplitude, width)"
            return p[0] * np.exp(-((x - center) / p[1])**2)

        def full_fit(x, *p):
            "p = (amplitude, width, center, background)"
            return p[3] + p[0] * np.exp(-((x - p[2]) / p[1]) ** 2)

        center_index = len(velocities) // 2
        dv = velocities[1] - velocities[0]
        coeffs = [powers[center_index], 4 * dv]  # initial guesses
        # estimate the amplitude
        coeff, var_matrix = curve_fit(
            just_amp_width, velocities, powers, p0=coeffs)

        # append coefficients to coeff for next fit
        new_coeff = [coeff[0], coeff[1], center, powers.mean()]
        final_coeff, var_matrix = curve_fit(
            full_fit, velocities, powers, p0=new_coeff
        )
        return dict(width=final_coeff[1],
                    amplitude=final_coeff[0],
                    center=final_coeff[2])

