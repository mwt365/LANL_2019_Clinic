# coding:utf-8
"""
::

  Author:  LANL Clinic 2019 --<lanl19@cs.hmc.edu>
  Purpose: Look through a dig file for timing fiducials
  Created: 11/10/19
"""

import os
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
from collections import OrderedDict

from ProcessingAlgorithms.preprocess.digfile import DigFile
from ProcessingAlgorithms.SaveFiles.save_as_dig_file import save_as_dig
from ProcessingAlgorithms.Fitting.moving_average import compress


class Fiducials:
    """
    This class looks for fiducial timing marks in a PDV DigFile.
    It assumes that these fiducials begin with a sharp change in the
    value of V(t), remain altered for a period of roughly
    notch_width = 1e-8 s, and then return to roughly the same
    level as before the notch.

    It uses two strategies to identify the fiducials. In the first
    pass, it smooths the data by averaging together consecutive values
    that span about one-eighth of the expected notch width. From the
    smoothed data, it computes a finite difference between
    smoothed values separated by about half the expected notch width.

    It then looks at the position of the minimum and maximum of the
    finite differences. If they are separated by a suitable time
    interval and if the voltage difference between them is large
    enough compared to the noise level of the diff values, the
    identification is deemed a success.

    It is then refined by attempting to fit a hyperbolic
    tangent function to the sudden drop at the start of the timing
    fiducial in the raw data, and another to the rise at the end.

    On success, the values are added to a pandas DataFrame with
    columns defined by Fiducials._columns (see below the comment).

    A second fiducial is sought roughly =window= from the first.
    Subsequent marks are sought respecting the spacing between successive
    fiducials.

    The property function *values* returns a pandas DataFrame with
    the information about the fiducials that were found.
    """

    _columns = ("t_start", "dt_start", "t_end", "dt_end", "width", "depth")

    def __init__(self, digfile: DigFile,
                 notch_width: float = 1e-8,
                 window: float = 100e-6):
        """
        ::
            Inputs:
                digfile:     a DigFile to analyze
                notch_width: expected duration of the fiducial (s)
                window:      expected duration between successive fiducials (s)

        """
        assert isinstance(digfile, DigFile)
        self.digfile = digfile
        self.width = notch_width  # rough time width of the fiducial
        self.window = window      # rough duration of a frame
        self.marks = pd.DataFrame(columns=self._columns)

        # Average over the following number of points to reduce noise
        self.average_over = int(self.width / 8 / digfile.dt)
        self.diff_shift = int(0.5 * notch_width / digfile.dt /
                              self.average_over)
        self.half_shift = self.diff_shift // 2

        self.voltage = None  # will hold a smoothed voltage trace
        self.diff = None    # will hold the difference array

        # Make a preliminary pass to look for the first fiducial
        self.find_fiducial(digfile.t0, window / 10 - digfile.t0)
        if self.width_test and self.snr_test:
            # Look for the next one
            self.find_fiducial(self.t_fiducial + self.window * 0.9,
                               0.2 * self.window)
            if self.width_test and self.snr_test and len(self.marks) == 2:
                self.propagate()

    @property
    def values(self):
        """Return a pandas DataFrame with the fiducials we successfully
        identified."""
        return self.marks

    def smooth(self, t_from: float, dt: float):
        """
        Prepare a smoothed version of the time data from t_from
        to t_from + dt, and a finite difference version from the
        smoothed data by subtracting "rolled" versions of the smoothed
        data, one rolled forward by self.half_shift, the other back
        by the same amount.

        No values are returned. Instead, the fields voltage, diff, t_start,
        and t_step are updated in self.
        """
        vals = self.digfile.values(t_from, t_from + dt)
        # To use efficient averaging, we need to use a multiple
        # of self.average_over.
        points = len(vals) // self.average_over
        avg = np.reshape(vals[:points * self.average_over],
                         (-1, self.average_over)).mean(axis=1)
        diff = np.roll(avg, -self.half_shift) - np.roll(avg, self.half_shift)
        diff[:self.half_shift] = 0
        diff[-self.half_shift:] = 0
        self.voltage = avg
        self.diff = diff
        self.t_start = t_from
        self.t_step = self.digfile.dt * self.average_over

    def find_fiducial(self, t_from: float, dt: float):
        """
        Look for a fiducial notch starting at t_from (s) over a temporal
        range of width dt (s). Start by preparing a smoothed version,
        using self.average_over to define how many consecutive values are combined
        to produce the smoothed representation. Then
        """
        self.smooth(t_from, dt)
        diff = self.diff
        try:
            low, hi = np.argmin(diff), np.argmax(diff)
            # a first sanity test is whether these are
            # suitably spaced
            t_ratio = (abs(hi - low) * self.t_step) / self.width
            # if the t_ratio is near 1, we pass
            self.width_test = t_ratio > 0.5 and t_ratio < 2

            # Let's try to quantify the signal-to-noise ratio
            if low > 100:
                sigma = np.std(diff[low - 100:low - 10])
            else:
                sigma = np.std(diff[low + 10:low + 100])

            if not self.width_test:
                # See if among the top 3 peaks and the top 3 troughs
                # there isn't a pair that has the suitable time separation

                ordering = np.argsort(self.diff)
                troughs, peaks = ordering[:15], ordering[-15:]
                for trough in troughs:
                    if diff[trough + 8] > sigma:
                        low = trough
                        hi = low + np.argmax(diff[low:low + 15])
                        self.width_test = True
                        break
                if not self.width_test:
                    for peak in peaks:
                        if diff[peak - 8] < -sigma:
                            low = peak - 15 + np.argmin(diff[peak - 15:peak])
                            hi = peak
                            self.width_test = True

            min_voltage, max_voltage = diff[low], diff[hi]
            snr = (max_voltage - min_voltage) / (2 * sigma)
            self.snr_test = snr > 2
            self.t_fiducial = (0.5 + min(low, hi)) * \
                self.t_step + self.t_start
            self.dt_fiducial = abs(hi - low) * self.t_step
            self.dv_fiducial = -min_voltage
            if self.width_test:
                # Is it now possible to refine the fiducial time?
                self.refine()
        except:
            pass

    def plot(self):
        """
        For debugging purposes, plot the data we're using
        to try to identify the fiducial
        """
        import matplotlib.pyplot as plt
        tvals = self.t_start + np.arange(len(self.diff)) * self.t_step
        fig = plt.figure(1)
        ax = fig.add_subplot(1, 1, 1)
        ax.clear()
        ax.plot(tvals * 1e6, self.diff, 'b-', alpha=0.5)
        # add points for the guessed fiducial
        t = self.t_fiducial * 1e6
        ax.plot([t, t], [0, -self.dv_fiducial], 'k-o')
        plt.show()

    def refine(self):
        """
        Attempt to better the value self.t_fiducial determined
        by the difference method by fitting tanh functions to the
        two edges. On success, add a row to the DataFrame at
        self.marks.
        """
        def edge(x, *params):
            "params = (t_halfway, v_left, v_right, width)"
            t_halfway, v_left, v_right, width = params
            arg = (x - t_halfway) / width
            amp = 0.5 * (v_right - v_left)
            avg = 0.5 * (v_right + v_left)
            return amp * np.tanh(arg) + avg

        # prepare a dictionary to use when adding the mark
        dic = dict(zip(self._columns, np.zeros(len(self._columns))))

        # Fetch a minimal range of data around the best guess
        dt = 4e-9
        vals = self.digfile.values(
            self.t_fiducial - dt, self.t_fiducial + dt)
        tvals = self.digfile.time_values(self.t_fiducial - dt,
                                         self.t_fiducial + dt)
        if len(self.marks) > 0:
            dt_start = self.marks['dt_start'].mean()
            width = self.marks['width'].mean()
            dt_end = self.marks['dt_end'].mean()
        else:
            dt_end = dt_start = 1e-10
            width = self.width

        params = (self.t_fiducial, vals[0], vals[-1], dt_start)
        try:
            coeffs, covar = curve_fit(edge, tvals, vals, p0=params)
            errs = np.sqrt(np.diag(covar))
            if np.nan not in covar and np.inf not in covar:
                # Update the values for t_fiducial and dv_fiducial
                self.t_fiducial = dic['t_start'] = coeffs[0]
                dic['dt_start'] = coeffs[3]
                depth = abs(coeffs[1] - coeffs[2])
                depth_err = np.sqrt(errs[1]**2 + errs[2]**2)
                if depth_err > depth:
                    return 0
                # Now look to find the upswing
                t = coeffs[0] + width
                vals = self.digfile.values(t - dt, t + dt)
                tvals = self.digfile.time_values(t - dt, t + dt)
                params = (t, coeffs[2], coeffs[1], dt_end)
                coeffs, covar = curve_fit(edge, tvals, vals, p0=params)
                errs = np.sqrt(np.diag(covar))
                depth_err = np.sqrt(errs[1]**2 + errs[2]**2)
                depth2 = abs(coeffs[2] - coeffs[1])
                if np.nan not in covar and np.inf not in covar \
                   and depth2 > depth_err:
                    dic['t_end'] = coeffs[0]
                    dic['dt_end'] = coeffs[3]
                    dic['depth'] = 0.5 * (depth + depth2)
                    dic['width'] = dic['t_end'] - dic['t_start']
                    self.marks = self.marks.append(dic, ignore_index=True)
        except Exception as eeps:
            print(f"Exception {eeps} refining fiducial for {self.digfile.title}")

    def propagate(self):
        """
        We have found two fiducial marks. Find the rest using the
        spacing we observe as a guideline.
        """
        spacing = self.marks.iloc[1]['t_start'] - \
            self.marks.iloc[0]['t_start']
        t = self.marks.iloc[1]['t_start']
        last_time = self.digfile.t_final - self.window
        try:
            while t < last_time:
                t += 0.99 * spacing
                self.find_fiducial(t, 0.02 * spacing)
                if self.width_test and self.snr_test:
                    t = self.t_fiducial
                else:
                    t += 0.01 * spacing
        except Exception as eeps:
            print(eeps)

    def split(self, basename=""):
        """
        Prepare a subdirectory filled with the segments that
        go from one timing fiducial to the next. If no basename
        is supplied, use the name of the source file before the
        .dig extension.
        """
        parent, file = os.path.split(self.digfile.path)
        folder, _ = os.path.splitext(file)
        if not basename:
            basename = "seg"
        # Make the directory
        home, filename = os.path.split(self.digfile.path)
        folder = os.path.join(parent, folder)
        if not os.path.exists(folder):
            os.makedirs(folder)
        times = self.marks['t_start'].to_numpy()
        # Prepare the top 512-byte header
        df = self.digfile
        text = open(df.path, 'rb').read(512).decode('ascii')
        text = text.replace("\000", "").strip()
        the_date = df.date

        kwargs = dict(
            dt=df.dt,
            initialTime=0,
            voltageMultiplier=df.dV,
            voltageOffset=df.V0
        )
        for n in range(len(times)):
            t_start = times[n]
            n_start = int((t_start - self.digfile.t0) / self.digfile.dt)
            try:
                t_stop = times[n + 1]
            except:
                t_stop = df.t_final
            heading = OrderedDict(
                Segment=n,
                n_start=n_start,
                t_start=f"{t_start:.6e}",
                t_stop=f"{t_stop:.6e}",
                date=the_date,
                header="\r\n" + text,
            )

            vals = df.raw_values(t_start, t_stop)
            name = f"{basename}{n:02d}.dig"
            save_as_dig(os.path.join(folder, name),
                        vals, df.data_format,
                        top_header=heading, date=the_date, **kwargs)
        return f"{len(times)} files written in {folder}"


def split_digfile(digfile: DigFile, times: np.ndarray, basename="", **kwargs):
    """
    Prepare a subdirectory filled with the segments that
    go from one timing fiducial to the next. If no basename
    is supplied, use the name of the source file before the
    .dig extension.
    """
    if not basename:
        basename = "seg"
    # Make the directory
    home, filename = os.path.split(digfile.path)
    folder = os.path.join(home, basename)
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Prepare the top 512-byte header
    text = digfile.header_text
    if len(text) > 512:
        text = text[:512]
    header = text
    kwargs = dict(
        dt=digfile.dt,
        initialTime=0,
        voltageMultiplier=digfile.dV,
        voltageOffset=digfile.V0
    )
    for n in range(len(times)):
        head = header.format(n)
        t_start = times[n]
        try:
            t_stop = times[n + 1]
        except:
            t_stop = digfile.t_final
        vals = digfile.raw_values(t_start, t_stop)
        name = f"{basename}_{n:02d}.dig"
        save_as_dig(os.path.join(folder, name),
                    vals, digfile.data_format,
                    top_header=head, **kwargs)


def downspike(digfile: DigFile, **kwargs):
    """Look for a downward spike followed by a roughly exponential relaxation
        towards the zero level with a time constant of roughly 0.5 µs in the range of
        times specified by t_range.
    """
    import matplotlib.pyplot as plt

    # Average consecutive samples over what time?
    average_over_time = kwargs.get('avg_time', 5.0e-8)
    average_over = int(average_over_time / digfile.dt)
    threshold = kwargs.get('threshold', 50)
    t_range = kwargs.get(
        't_range', (digfile.t0, min(digfile.t_final, 100e-6)))
    rawt, rawv = digfile.microseconds(*t_range), digfile.values(*t_range)
    tvals, vvals = compress(average_over, rawt, rawv)
    plot = kwargs.get('plot', False)

    try:
        lead_points = kwargs.get('lead_points', 100)
        amin = np.argmin(vvals)
        assert amin > lead_points, "The spike appears too close to the start of the time range"
        noise = np.std(vvals[0:lead_points])

        # How far below the noise level do we need to go
        # to detect a downward spike?
        threshold *= -noise
        # Look for the first point below threshold
        spike = lead_points + np.argmax(vvals[lead_points:] < threshold)

        if plot:
            axes = plt.gca()
            axes.clear()
            axes.plot(rawt, rawv, 'y-', alpha=0.25)  # plot raw data
            axes.plot(tvals, vvals, 'g-')  # plot smoothed version
            # plot the threshold for a downspike
            axes.plot([tvals[0], tvals[-1]],
                      [threshold, threshold], 'k--', alpha=0.5)
            axes.plot([tvals[spike]], [vvals[spike]], 'bo', alpha=0.5)

        def myline(t, *params):
            "params = t0, m"
            return params[1] * (params[0] - t)

        def myexpo(t, *params):
            "params = y0, tau"
            t0 = t[0]
            return params[0] * np.exp((t0 - t) / params[1])

        params, covars = curve_fit(
            myline, tvals[spike - 5:spike + 5], vvals[spike - 5:spike + 5], p0=(tvals[spike - 5], 1000.))
        t0 = params[0]

        # Identify the next minimum
        onemicro = int(1e-6 / average_over_time)

        minpos = spike + np.argmin(vvals[spike:spike + 2 * onemicro])

        # error check?
        exparams, expcovars = curve_fit(
            myexpo, tvals[minpos:minpos +
                          onemicro], vvals[minpos:minpos + onemicro],
            p0=(vvals[minpos], 0.5)
        )
        tau = exparams[1]

        if plot:
            axes.plot([t0], [0.0], 'r*')  # where we think the spike starts
            axes.plot(tvals[minpos:minpos + onemicro],
                      myexpo(tvals[minpos:minpos + onemicro], *exparams), 'r--')
            axes.set_xlim(tvals[spike - 100], tvals[minpos + 2 * onemicro])
            axes.set_title(digfile.filename, usetex=False)
            axes.set_xlabel("$t$")
            plt.show()
        assert tau < 2

        return t0
    except Exception as eeps:
        print(eeps)


if __name__ == "__main__":

    os.chdir('../../../dig')
    df = DigFile('GEN1_CHAN1TEKBAK001.dig')
    fids = Fiducials(df)
    print(fids.values)

    #candidates = DigFile.all_dig_files()
    # for filename in candidates:
    #df = DigFile(filename)
    #spike = downspike(df, plot=True)
    # ans = f"{spike:.5f} µs" if spike else "-"
    #print("{0: >.24s}  {1}".format(filename, ans))

    ## fid = Fiducials(df)
    # print(fid.values)
    ## df = DigFile('../dig/GEN3_CHANNEL1KEY001')
    # if False:
    #fid = Fiducials(df)
    # print(fid.values)
