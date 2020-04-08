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
from scipy.interpolate import InterpolatedUnivariateSpline as ius


def smooth(xvals, n=10, forward=True):
    """Provide an exponential smoothing of the values in xvals,
    yielding a single value representing the final smoothed value
    in the array. If forward is False, proceed in reverse order."""

    if not forward:
        xvals = np.flip(xvals)
    fraction = 1 - 1 / n
    num_to_use = min(n, len(xvals))
    s = xvals[-num_to_use]
    for x in xvals[-num_to_use:]:
        s *= fraction
        s += (1 - fraction) * x
    return s


def extrapolate(xvals, yvals, x, grouping=5, order=3):
    """Given equal-length arrays xvals and yvals, with the
    xvals in ascending order, produce an interpolated or extrapolated
    estimate for y at corresponding x. If there are enough points to
    use grouping to smooth out noise in the yvals, average over
    consecutive points up to groups of size grouping.
    """

    assert len(xvals) == len(yvals)
    n = len(xvals)

    if n == 1:
        return yvals[0]

    if n == 0:
        raise IndexError

    # Do we have enough points to form (order+1) groups of size
    # grouping?
    while True:
        points = grouping * (order + 1)
        if points <= n:
            break
        if order > 1:
            order -= 1
        elif grouping > 1:
            grouping -= 1
        else:
            break

    if grouping > 1:
        pts = grouping * (order + 1)
        for array in (xvals, yvals):
            if x <= xvals[0]:
                tmp = array[:pts]
            else:
                tmp = array[-pts:]
            tmp = np.mean(
                np.reshape(tmp, (grouping, -1)), axis=1)
            if array == xvals:
                xvals = tmp
            else:
                yvals = tmp

    try:
        spline = ius(xvals, yvals, k=order)
        return float(spline(x))
    except Exception as eeps:
        print(eeps)
    return yvals[0]  # ???


class Follower:
    """
    Given a spectrogram and a starting point (t, v) --
    or a list of such points -- a follower
    looks for a quasicontinuous local maximum through the spectrogram.
    This base class handles storage of the spectrogram reference, the
    starting point, and a span value describing the width of the
    neighborhood to search, centered on the previous found maximum.

    It also holds a results dictionary with several obligatory
    fields, to which a subclass may add. The required fields are

        vi_span: the range of point indices used
        time:                the times found (s)
        t_index:          the index of the time columns
        peak_v:       the peak velocities
        peak_int:      the intensity at the peak

    """

    def __init__(self, spectrogram, start_point, span=80):
        assert isinstance(spectrogram, Spectrogram)
        assert isinstance(span, int)
        self.spectrogram = spectrogram

        # Now establish storage for intermediate results and
        # state. t_index is the index into the spectrogram.intensity
        # along the time axis.

        self.results = dict(
            vi_span=[],  # the range of points used to look for the peak
            time=[],                 # the time in s
            t_index=[],           # the corresponding point index in the time dimension
            peak_v=[],        # the peak velocity
            v_index=[],       # corresponding index
            peak_int=[]        # the peak intensity
        )
        # for convenience, install pointers to useful fields in spectrogram
        self.velocity = self.spectrogram.velocity
        self.time = self.spectrogram.time
        self.intensity = self.spectrogram.intensity

        # convert to a numpy array so we can inspect dimensions
        stpt = np.asarray(start_point)
        if stpt.size == 2:
            # we have a single start point
            self.t_start = start_point[0]
            self.v_start = start_point[1]
        else:
            self.t_start = stpt[0, 0]
            self.v_start = stpt[0, 1]
            for t, v in start_point:
                t_index = self.spectrogram._time_to_index(t)
                self.add_point(t_index, v)

        self.t_index = spectrogram._time_to_index(self.t_start)
        self.span = span

    def add_point(self, t_index, v, span=None):
        """Add this point to the results, making sure to keep the results
        array sorted by time.
        """
        r = self.results
        v_index = self.spectrogram._velocity_to_index(v)
        dic = dict(
            time=self.time[t_index],
            t_index=t_index,
            peak_v=v,
            peak_int=self.spectrogram.intensity[v_index, t_index],
            vi_span=span,
            v_index=v_index
        )

        if len(r['time']) and t_index < r['t_index'][0]:
            # We need to insert at the beginning
            for k, v in dic.items():
                r[k].insert(0, v)
        else:
            for k, v in dic.items():
                r[k].append(v)

    @property
    def v_of_t(self):
        "A convenience function for plotting; returns arrays for time and velocity"
        t = np.array(self.results['time'])
        v = np.array(self.results['peak_v'])
        return t, v

    def guess_range(self, t_index):
        """
        Attempt to guess where the center of the peak should be for time
        index t_index, using the best information available. If we have no
        data yet, use t_start and v_start
        """

        r = self.results
        times, vels = r['time'], r['peak_v']
        if len(r['time']) == 0:
            v_guess = self.v_start
        else:
            v_guess = extrapolate(
                times, vels, self.spectrogram.time[t_index])

        def bound(x):
            if x < 0:
                return 0
            if x >= len(self.spectrogram.velocity):
                return len(self.spectrogram.velocity) - 1
            return x
        v_index = self.spectrogram._velocity_to_index(v_guess)
        start_index = bound(v_index - self.span)
        end_index = bound(v_index + self.span)

        # If we have some prior information, make sure we adjust to include
        # prior information.
        if len(r['time']) > 0:
            dt = np.abs(np.array(r['t_index']) - t_index)
            last_index = np.argsort(dt)[0]  # Find the closest index
            last_v_index = r['v_index'][last_index]
            start_index = min(last_v_index - 5, start_index)
            end_index = max(last_v_index + 5, end_index)
        return (start_index, end_index)

    def guess_intensity(self, t_index):
        """Attempt to guess the expected intensity based on extrapolation"""
        r = self.results
        times = r['t_index']
        intensity = r['peak_int']
        n = len(times)
        if n == 0:
            return 0
        if n == 1:
            return intensity[0]
        else:
            return smooth(intensity, forward=t_index >= times[-1])

    def guess_intensity_old(self, t_index):
        """Attempt to guess the expected intensity based on extrapolation"""
        r = self.results
        times = r['t_index']
        intensity = r['peak_int']
        n = len(times)
        if n == 0:
            return 0
        if n == 1:
            return intensity[0]
        else:
            return extrapolate(times, intensity, t_index)

    def data_range(self, n=-1):
        """
        Fetch the span of velocities and intensities to use for fitting.
        The default value of n (-1) indicates the last available point
        from the results dictionary. Earlier points are possible.
        """
        if len(self.results['peak_v']) > 0:
            last_v = self.results['peak_v'][n]
        else:
            last_v = self.v_start
        v_index = self.spectrogram._velocity_to_index(last_v)
        start_index = max(0, v_index - self.span)
        end_index = min(v_index + self.span,
                        len(self.spectrogram.velocity))
        return (start_index, end_index)

    def data(self, n=-1):
        #  old way: start_index, end_index = self.data_range(n)
        start_index, end_index = self.guess_range(n)
        velocities = self.velocity[start_index:end_index]
        intensities = self.intensity[start_index:end_index, self.t_index]
        return velocities, self.spectrogram.power(intensities), start_index, end_index

    def hood(self, **kwargs):
        """
        Return a FollowerHood describing the data and results at
        point n of the follower.
        To indicate which hood to return, the possible kwargs are
            n = index of the point in the follower
            t_index = t_index of the point
            t = time (in seconds)
        """
        if 'n' in kwargs:
            n = kwargs['n']
        elif 't_index' in kwargs:
            t_index = kwargs['t_index']
            n = self.results['t_index'].find(t_index)
        elif 't' in kwargs:
            t = kwargs['t']
            t_index = self.spectrogram._time_to_index(t)
            n = self.results['t_index'].find(t_index)
        else:
            raise Exception("I can't identify the hood")
        return FollowHood(self, n)

    @property
    def neighborhoods(self):
        if not hasattr(self, '_neighborhoods'):
            self._neighborhoods = [
                self.hood(n=j) for j in range(len(self.results['time']))
            ]
        return self._neighborhoods

    @property
    def frame(self):
        """
        Return a pandas DataFrame holding the results of this
        following expedition, with an index of the times
        converted to microseconds.
        """
        microseconds = np.array(self.results['time']) * 1e6
        return pd.DataFrame(self.results, index=microseconds)

    @property
    def full_frame(self):
        """
        Return a pandas DataFrame holding the results of this
        following expedition, with an index of the times
        converted to microseconds, and including all the
        information obtained from calling hood on each point.
        """
        df = pd.DataFrame(self.results)
        hoods = self.neighborhoods

        # add columns obtained from the hoods
        df['v_index'] = [h.v_index for h in hoods]
        df['m_center'] = [h.moment['center'] for h in hoods]
        df['m_width'] = [h.moment['std_err'] for h in hoods]
        df['m_bgnd'] = [h.moment['background'] for h in hoods]
        df['g_center'] = [h.gaussian.center for h in hoods]
        df['g_width'] = [h.gaussian.width for h in hoods]
        df['g_bgnd'] = [h.gaussian.background for h in hoods]
        df['g_int'] = [h.gaussian.amplitude for h in hoods]
        # return the frame with the index set to time in microseconds
        df.index = df['time'] * 1e6
        # Now we need to set the formatting with
        # df.style.format(dictionary of callables)
        return df

    @property
    def pandas_format(self):
        def d1(x): return f"{x:.1f}"

        def d2(x): return f"{x:.2f}"

        formats = {x: d2 for x in
                   "peak_v,peak_int,m_center,m_width,m_bgnd,g_center,g_width,g_bgnd,g_int".split(',')}
        formats['time'] = lambda x: f"{x * 1e6:.2f} µs"
        return formats

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
        res['velocity_uncertainty'] = np.zeros(len(res['peak_v']))
        for n in range(len(res['t_index'])):
            fit_res = self.estimate_width(n)
            res['peak_v'][n] = fit_res['center']
            res['velocity_uncertainty'][n] = fit_res['width']
            res['peak_int'][n] = fit_res['amplitude']

    def estimate_width(self, n, neighborhood=32):
        """
        Estimate the gaussian width of a peak
        """
        res = self.results
        t_index = res['t_index'][n]
        v_peak = res['peak_v'][n]
        v_peak_index = self.spectrogram._velocity_to_index(v_peak)

        hoods, means, stdevs = [], [], []
        hood = neighborhood
        while hood > 1:
            n_low = max(0, v_peak_index - hood)
            n_high = min(v_peak_index + hood + 1, len(self.velocity))
            # fetch the true power values for this column of the spectrogram
            power = self.spectrogram.power(
                self.intensity[n_low:n_high, t_index])
            velocities = self.velocity[n_low:n_high]
            if hood == neighborhood:
                res = self.fit_gaussian(
                    self.velocity[n_low:n_high], power, v_peak)
                res['power'] = power
                res['indices'] = (n_low, n_high)
            mean, stdev = self.moments(velocities, power)
            hoods.append(hood)
            means.append(mean)
            stdevs.append(stdev)
            hood = hood // 2
        print(stdevs)
        #

        return res

    def moments(self, x, y):
        """
        Give an array x with equally spaced points and an
        array y holding corresponding intensities, with a
        peak allegedly near the middle, use the first
        moment to estimate the center and the second moment
        to estimate the width of the peak
        """
        # Let's attempt to remove noise by axing points
        # below 5% of the peak
        threshold = y.max() * 0.05
        clean = np.array(y)
        clean[clean < threshold] = 0
        zero = clean.sum()
        one = np.sum(x * clean)
        two = np.sum(x * x * clean)
        mean = one / zero
        var = two / zero - mean**2
        stdev = np.sqrt(var)
        return (mean, stdev)

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


class FollowHood(object):
    """
    All information about the neighborhood around a point
    identified on a curve by a follower:

      - time       in seconds
      - t_index    column of the spectrogram
      - peak_v     in m/s
      - v_index    row of the spectrogram
      - peak_int   intensity at the peak
      - vi_span    rows used to look for peak
      - velocity   velocity values of above
      - intensity  intensity values of above
      - moment     {'center', 'variance', 'std_dev', 'std_err', 'background'}
      - gaussian   {background, amplitude, center, width}
    """

    def __init__(self, follower: Follower, pt: int):
        """

        """
        from ProcessingAlgorithms.Fitting.gaussian import Gaussian
        from ProcessingAlgorithms.Fitting.moments import moment

        res = follower.results
        assert pt >= 0 and pt < len(res['time'])

        self.follower = follower
        self.time = res['time'][pt]
        self.t_index = res['t_index'][pt]
        self.peak_v = res['peak_v'][pt]
        self.vi_span = res['vi_span'][pt]
        self.peak_int = res['peak_int'][pt]

        # Now fetch the raw points from the spectrogram
        sg = follower.spectrogram
        self.v_index = sg._velocity_to_index(self.peak_v)
        vfrom, vto = self.vi_span

        # The velocity array and intensity array
        self.velocity = sg.velocity[vfrom:vto]
        self.intensity = sg.power(sg.intensity[vfrom:vto, self.t_index])

        # Compute width from moments
        # moment is a dictionary with fields 'center' and 'std_dev'
        # We want to narrow the range of points used for computing moments
        # to those around the central peak
        def bound(x):
            x = max(x, 0)
            return min(x, len(self.velocity) - 1)

        hood = 6
        m = dict(std_dev=np.nan)
        while np.isnan(m['std_dev']):
            pfrom, pto = [bound(x + self.v_index - vfrom)
                          for x in (-hood, hood)]
            m = moment(self.velocity[pfrom:pto], self.intensity[pfrom:pto])
            hood += 4
        self.moment = m

        # Fit a gaussian
        self.gaussian = Gaussian(
            self.velocity, self.intensity,
            center=self.peak_v,
            width=self.moment['std_dev']
        )

    def plot_gaussian(self, ax, **kwargs):
        """
        Compute points for a smooth gaussian curve and
        plot it on axes ax.
        """
        g = self.gaussian
        if g.valid:
            sig = g.width
            middle = np.arange(g.center - 3 * sig,
                               g.center + 3 * sig, 0.1 * sig)
            left = np.arange(g.center - 10 * sig, g.center - 3 * sig, sig)
            right = np.arange(g.center + 3 * sig, g.center + 10 * sig, sig)
            v = np.concatenate((left, middle, right))
            i = g(v)
            ax.plot(v, i, 'b-', alpha=0.5, label='gaussian')

    def plot_all(self, ax, **kwargs):
        """
        Show all the "hood" information on a plot on axes.
        The caller should clear the axes first, if desired, with
        ax.clear().
        """
        ax.plot(self.velocity, self.intensity,
                'ko', alpha=0.5, label='data')
        # plot the background level used for the moment calculation
        bg = self.moment['background']
        ax.plot([self.velocity[0], self.velocity[-1]],
                [bg, bg], 'r-', label='bgnd')
        # show the center and widths from the moment calculation
        tallest = np.max(self.intensity)
        ax.plot([self.moment['center'] + x * self.moment['std_err'] for x in
                 (-1, 0, 1)], 0.5 * tallest * np.ones(3), 'r.', label='moments')
        # show the gaussian
        self.plot_gaussian(ax)
        vcenter, width = self.peak_v, self.moment['std_err']
        ax.set_xlim(vcenter - 12 * width, vcenter + 12 * width)

        xlabel = kwargs.get('xlabel')
        ylabel = kwargs.get('ylabel')
        title = kwargs.get('title')
        legend = kwargs.get('legend', True)

        if xlabel:
            ax.set_xlabel("$v$ (m/s)")
        if ylabel:
            ax.set_ylabel("Intensity")
        if title:
            ax.set_title(title)
        if legend:
            ax.legend()
        return tallest
