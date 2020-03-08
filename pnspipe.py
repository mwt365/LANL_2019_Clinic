#!/usr/bin/env python3
# coding:utf-8
"""
  Author:  LANL Clinic 2019 --<lanl19@cs.hmc.edu>
  Purpose: Process a set of .dig files
  Created: 02/21/20
"""
import datetime
import os
import sys
from time import time
import inspect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotter import COLORMAPS
DEFMAP = '3w_gby'

from ProcessingAlgorithms.preprocess.digfile import DigFile


def all_pipe_functions():
    funcs = []
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isfunction(obj):
            sig = list(inspect.signature(obj).parameters.keys())
            if sig and sig[0] == 'pipe' and len(sig) == 2:
                funcs.append(obj)
    return funcs


def describe_pipe_functions():
    """
    List all defined pipe functions, with their docstrings.
    """
    thestrs = [f"{x.__name__}\n{x.__doc__}" for x in all_pipe_functions()]
    return "\n\n".join(thestrs)


class PNSPipe:
    """

    """

    def __init__(self, filename, orders, **kwargs):
        """
        On entry we assume that we are in the directory where
        the report and output should be written, in a folder
        hierarchy that mirrors the structure to the .dig file.
        """
        self.home = os.getcwd()
        self.filename = filename
        self.orders = orders
        self.df = DigFile(filename)
        self.rel_dir = os.path.join(self.df.rel_dir, self.df.basename)
        self.output_dir = os.path.join(self.home, self.rel_dir)

        # Make sure that this directory exists
        os.makedirs(self.rel_dir, exist_ok=True)
        print(f"Making directories {self.rel_dir}")
        # Open a log file
        self.logfile = open(
            os.path.join(self.output_dir, 'info.txt'),
            'w', buffering=1)

        now = datetime.datetime.now()
        self.log(f"{self.filename} log, {now.strftime('%a %d %b %Y at %H:%M:%S')}")
        self.spectrogram = None
        self.baselines = []
        self.gaps = []
        self.jumpoff = None
        self.probe_destruction = None
        self.followers = []
        self.signals = []
        self.pandas_format = dict(
            time=lambda x: f"{x*1e6:.3f}",
            peak_velocity=self.onedigit,
            peak_intensity=self.onedigit,
            gaussian_center=self.onedigit,
            gaussian_width=self.onedigit,
            gaussian_background=self.twodigit,
            gaussian_intensity=self.twodigit,
            moment_center=self.onedigit,
            moment_width=self.onedigit,
            moment_background=self.twodigit,
            amplitude=self.onedigit,
            dcenter=self.twodigit,
        )

        # How do we specify spectrogram parameters?
        # If the first command is not a specification for computing
        # a spectrogram, use the defaults

        if orders[0][0] != make_spectrogram:
            orders.insert(0, (make_spectrogram, {}))

        for order in orders:
            routine, kwargs = order
            self.start(routine, kwargs)
            print(f"{routine.__name__} ...", end="", flush=True)
            routine(self, **kwargs)
            print(self.end())

    def onedigit(self, val):
        return f"{val:.1f}"

    def twodigit(self, val):
        return f"{val:.2f}"

    def log(self, x):
        self.logfile.write(x + '\n')

    def open(self, message: str, kwargs: dict):
        """
        Start a frame with label message. If kwargs is not
        empty, include the parameters in the dictionary.
        """
        self.log(f"<<<<< {message}")
        if kwargs:
            for k, v in kwargs.items():
                self.log(f"+ {k} = {v}")

    def close(self, message: str):
        msg = f">>>>> {message}"
        self.log(msg + "\n")
        return msg

    def start(self, routine, kwargs):
        """
        Call this on entry to get the caller's information
        added to the log and to start the timer for the
        routine.
        """
        self.t0 = time()
        self.open(routine.__name__, kwargs)

    def end(self):
        """
        Call this on exit
        """
        return self.close(self._timestr(time() - self.t0))

    def _timestr(self, dt):
        if dt > 0.1:
            unit = 's'
        elif dt > 0.001:
            dt *= 1000
            unit = 'ms'
        elif dt > 1e-6:
            dt *= 1e6
            unit = 'µs'
        return f"{dt:.2f} {unit}"

    def __del__(self):
        try:
            # print("I'm in the destructor")
            self.logfile.flush()
            self.logfile.close()
            os.chdir(self.home)
            print(os.getcwd())
        except:
            pass


def make_spectrogram(pipe: PNSPipe, **kwargs):
    """
    Compute a spectrogram using the passed kwargs or
    built-in defaults and set the pipe.spectrogram field.
    Default values are:
      wavelength = 1.55e-6
      points_per_spectrum = 8192
      overlap = 0.5
      window_function = None
      form = 'power'
    """
    from spectrogram import Spectrogram

    defaults = dict(
        t_start=kwargs.get('t_start'),
        ending=kwargs.get('ending'),
        wavelength=kwargs.get('wavelength', 1.55e-6),
        points_per_spectrum=kwargs.get('points_per_spectrum', 8192),
        overlap=kwargs.get('overlap', 0.5),
        window_function=kwargs.get('window_function'),
        form=kwargs.get('form', 'power'),
    )
    allargs = {**defaults, **kwargs}
    # list the arguments we use in the log
    for k, v in allargs.items():
        pipe.log(f"+ {k} = {v}")

    pipe.spectrogram = Spectrogram(pipe.df, **allargs)


def find_destruction(pipe: PNSPipe, **kwargs):
    """
    Look for a signature of probe destruction using the
    vertical squash method.
    """
    sg = pipe.spectrogram
    t_blast = sg.vertical_spike()
    if t_blast:
        pipe.probe_destruction = t_blast
        pipe.log(f"probe destruction at {t_blast*1e6:.2f} µs")


def find_baselines(pipe: PNSPipe, **kwargs):
    """
    Compute baselines for pipe.spectrogram using the
    baselines_by_squash method. If baseline_limit is
    passed as a keyword argument, only keep baseline values
    that are larger than this value (which should be between
    0 and 1).
    """
    from baselines import baselines_by_squash as bline
    peaks, widths, heights = bline(pipe.spectrogram)
    baseline_limit = kwargs.get('baseline_limit', 0.01)
    pipe.baselines = peaks[heights > baseline_limit]
    pipe.log(f"Baselines > {baseline_limit*100}%: {pipe.baselines}")


def find_signal(pipe: PNSPipe, **kwargs):
    """
    Start at t_start and look for a peak above the baseline
    """
    from ProcessingAlgorithms.SignalExtraction.peak_follower import signal_finder

    t_start = float(kwargs.get('t_start', 2e-5))
    blines = sorted(pipe.baselines)
    sg = pipe.spectrogram
    guess = signal_finder(sg, blines, t_start)
    pipe.signal_guess = (t_start, guess)
    pipe.log(f"find_signal guesses signal is at {pipe.signal_guess}")


def find_signal_old(pipe: PNSPipe, **kwargs):
    """
    Start at t_start and look for a peak above the baseline
    """
    from scipy.signal import find_peaks

    # guess 20 µs if no info provided
    ts = float(kwargs.get('t_start', 2e-5))
    sg = pipe.spectrogram
    t_index = sg._time_to_index(ts)
    # average over a small time neighborhood
    spectrum = np.sum(sg.intensity[:, t_index - 2:t_index + 2], axis=1)
    peaks, properties = find_peaks(spectrum, height=0.001 * spectrum.max(),
                                   distance=40)
    try:
        heights = properties['peak_heights']
        # produce an ordering of the peaks from high to low
        ordering = np.flip(np.argsort(heights))
        peak_index = peaks[ordering]
        peaks = sg.velocity[peak_index]
        hts = heights[ordering]

        # normalize to largest peak of 1 (e.g., the baseline)
        hts = hts / hts[0]

        blines = sorted(pipe.baselines)
        # filter out any peaks on or below the baseline
        # We'll slide up a couple of velocity steps, just to be sure
        main_baseline = blines[0] + 5 * sg.dv
        hts = hts[peaks > main_baseline]
        peaks = peaks[peaks > main_baseline]
        # Our guess for the signal is now at the first peak

        # Our guess for the signal is now at the first peak
        pipe.signal_guess = (ts, peaks[0])
        pipe.log(f"find_signal guesses signal is at {pipe.signal_guess}")
    except:
        pipe.signal_guess = None
        pipe.log("find_signal could not find a signal")


def follow_signal(pipe: PNSPipe, **kwargs):
    """
    Using the peak_follower, look first look left, then right.
    Optional kwargs:
      plot = extension of the plot file type to save, or None;
             default is 'pdf'
      image = (boolean) produce a spectrogram image with superimposed follower

    """
    from ProcessingAlgorithms.SignalExtraction.peak_follower import PeakFollower
    if not pipe.signal_guess:
        pipe.log("No signal_guess so not following a signal.")
        return

    plot = kwargs.get('plot', 'pdf')
    image = kwargs.get('image', False)

    follower = PeakFollower(
        pipe.spectrogram,
        pipe.signal_guess,
        direction=-1)
    pipe.followers.append(follower)

    follower.run()
    follower.reverse()
    follower.run()

    signal = follower.full_frame  # pd.DataFrame(follower.results)
    signal.style.format(pipe.pandas_format)
    pipe.signals.append(signal)

    pipe.log(f"follow_signal generated {len(signal)} points")
    pipe.log(signal.to_string(
        formatters=pipe.pandas_format, sparsify=False))

    if plot:
        plt.clf()
        plt.plot(signal['time'] * 1e6, signal['peak_velocity'])
        plt.xlabel('$t~(\mu \mathrm{s})$')
        plt.ylabel('$v~(\mathrm{m/s})$')
        plt.title(pipe.df.title, usetex=False)
        plt.savefig(os.path.join(pipe.output_dir, 'follower.' + plot))

    if image:
        # Produce a plot that superimposes the follower on the
        # spectrogram, which should be clipped below by the baseline
        # and to the right by probe_destruction, if it exists
        plt.clf()  # clear current figure
        sg = pipe.spectrogram
        t_range = (
            sg.time[0], pipe.probe_destruction if pipe.probe_destruction else sg.time[-1])
        r = follower.results
        if r['peak_velocity']:
            top = np.max(r['peak_velocity']) + 400.0
        else:
            top = 7000  # this is terrible!

        v_range = pipe.baselines[0], top
        time, velocity, intensity = sg.slice(t_range, v_range)
        max_intensity = np.max(intensity)
        image = plt.pcolormesh(time * 1e6, velocity,
                               np.log10(intensity + 0.0001 * max_intensity) * 20)
        image.set_cmap(COLORMAPS[DEFMAP])
        ax = plt.gca()
        ax.plot(signal['time'] * 1e6,
                signal['peak_velocity'], 'r.', alpha=0.05)
        plt.colorbar(image, ax=ax, fraction=0.08)
        plt.xlabel(r'$t$ ($\mu$s)')
        plt.ylabel('$v$ (m/s)')
        plt.title(pipe.df.title, usetex=False)
        plt.savefig(os.path.join(pipe.output_dir, 'spectrogram.png'))


def show_discrepancies(pipe: PNSPipe, **kwargs):
    """
    Run through the followers and flag those for which
    there is an appreciable discrepancy between the results
    of a gaussian fit and the moments estimation of the
    peak width.

    kwargs:
      - sigmas: (1.5), the minimum discrepancy to trigger report
      - neighborhood:

    Run through signals (which are pandas DataFrames) and
    add columns for gaussian parameters: (center, width,
    amplitude, and dcenter), where dcenter represents the
    difference between the peak found by the peak follower
    and the center found by the Gaussian fit.

    The optional kwarg neighborhood specifies the number of
    points to include on either side of the peak found by the
    follower. If no value is given, a default of 10 points on
    either side of the peak is used.
    """

    neighborhood = kwargs.get('neighborhood', 10)
    sigmas = kwargs.get('sigmas', 2)  # discrepancy greater than this
    # will be reported
    widths = kwargs.get('widths', False)
    peak_gauss = kwargs.get('peak_gauss', False)
    peak_moment = kwargs.get('peak_moment', False)

    all_discrepancies = set()
    for signal, follower in zip(pipe.signals, pipe.followers):
        # identify the rows in the signal DataFrame where the
        # gaussian_width and the moment_width disagree

        rows = signal.copy().reset_index(drop=True)
        if widths:
            Crit1 = rows.gaussian_width > sigmas * rows.moment_width
            Crit2 = rows.gaussian_width * sigmas < rows.moment_width
            width_discrepancies = rows[Crit1 | Crit2].index.to_numpy()
            all_discrepancies.union(width_discrepancies)
        if peak_gauss:
            Crit3 = np.abs(rows.peak_velocity - rows.gaussian_center) / \
                rows.gaussian_width > sigmas
            peak_gauss_discrepancies = rows[Crit3].index.to_numpy()
            all_discrepancies.union(peak_gauss_discrepancies)
        if peak_moment:
            Crit4 = np.abs(rows.peak_velocity -
                           rows.moment_center) / rows.moment_width > sigmas
            peak_moment_discrepancies = rows[Crit4].index.to_numpy()
            all_discrepancies.union(peak_moment_discrepancies)

        # combine all these possibilities
        all_suspects = sorted(list(all_discrepancies))
        if all_suspects:
            hoods = follower.neighborhoods
            for suspect in all_suspects:
                hood = hoods[suspect]
                fig, ax = plt.subplots(1, 1)
                hood.plot_all(ax[0], xlabel=True, ylabel=True)
                fig.savefig(os.path.join(pipe.output_dir, f'bad{suspect}.png'))
                fig.close()


def gaussian_fit(pipe: PNSPipe, **kwargs):
    """
    oops
    """
    for signal in pipe.signals:
        # First add the requisite columns
        blanks = np.zeros(len(signal)) + np.nan
        signal['center'] = blanks
        signal['width'] = blanks
        signal['amplitude'] = blanks
        signal['dcenter'] = blanks
        signal['mean'] = blanks
        signal['sigma'] = blanks

        for n in range(len(signal)):
            row = signal.iloc[n]
            t_index = row['time_index']
            vpeak = sg._velocity_to_index(row['velocity'])
            # vfrom, vto = row['velocity_index_span']
            vfrom, vto = vpeak - neighborhood, vpeak + neighborhood
            powers = sg.intensity[vfrom:vto, t_index]
            speeds = sg.velocity[vfrom:vto]
            mom = moment(speeds, powers)
            signal.loc[n, 'mean'] = mom['center']
            signal.loc[n, 'sigma'] = mom['std_dev']
            gus = Gaussian(
                speeds, powers,
                center=row['velocity'],
                width=sg.velocity[2] - sg.velocity[0]
            )
            if gus.valid:
                signal.loc[n, 'center'] = gus.center
                signal.loc[n, 'width'] = gus.width
                signal.loc[n, 'amplitude'] = gus.amplitude
                diff = gus.center - signal.loc[n, 'velocity']
                signal.loc[n, 'dcenter'] = diff
                discrepancy = abs(diff / gus.width)
                if discrepancy > sigmas:
                    # The difference between the peak follower and the gaussian
                    # fit was more than 1 value of the width.
                    # Let's print out a plot to show what's going on
                    plt.clf()
                    vmin, vmax = sg.velocity[vfrom], sg.velocity[vto]
                    # make sure we have all the freshest values
                    row = signal.iloc[n]

                    plt.plot([row['velocity'], ],
                             [row['intensity'], ], 'k*')
                    plt.plot([row['mean'] + n * row['sigma'] for n in (-1, 0, 1)],
                             [row['intensity'] for x in (-1, 0, 1)],
                             'gs')
                    plt.plot(speeds, powers, 'r.')
                    vels = np.linspace(gus.center - 6 * gus.width,
                                       gus.center + 6 * gus.width, 100)
                    plt.plot(vels, gus(vels), 'b-', alpha=0.5)
                    tval = f"${pipe.spectrogram.time[t_index]*1e6:.2f}"
                    plt.title(tval + "$~ µs")
                    plt.xlabel("Velocity (m/s)")
                    plt.ylabel(r"Intensity")
                    plt.xlim(vmin, vmax)
                    plt.savefig(os.path.join(pipe.output_dir, f'bad{t_index}.pdf'))
                    plt.close()
                    with open(os.path.join(pipe.output_dir, f'bad{t_index}.txt'), 'w') as f:
                        f.write(pd.DataFrame(
                            {'power': powers, }, index=speeds).to_csv())

    for signal in pipe.signals:
        pipe.log(signal.to_string(
            formatters=pipe.pandas_format, sparsify=False))
        pipe.log("\n\n")
        plt.clf()
        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(6, 6))
        top, middle, bottom = axes
        top.semilogy(signal.time * 1e6, signal.intensity)
        top.set_ylabel('Intensity')
        top.set_title(pipe.df.basename)
        middle.plot(signal.time * 1e6, signal.velocity)
        middle.set_ylabel(r'$v~(\mathrm{m/s})$')
        bottom.errorbar(signal['time'] * 1e6,
                        signal.dcenter, yerr=signal.width, fmt='b.',
                        markersize=1.0, lw=0.5)
        bottom.set_xlabel(r'$t~(\mu \mathrm{s})$')
        bottom.set_ylabel(r'$\delta v~(\mathrm{m/s})$')
        plt.savefig(os.path.join(pipe.output_dir, 'gauss.pdf'))


def decode_arg(x):
    """
    Attempt to convert the value x to an int, a float, or a string
    """
    x = x.strip()
    try:
        return int(x)
    except:
        try:
            return float(x)
        except:
            return x


if __name__ == '__main__':
    # sys.path.insert(0, '../')
    import argparse
    import re
    import shutil

    curdir = os.getcwd()
    parser = argparse.ArgumentParser(
        description='Run a pipe to process .dig files',
        prog="pipe",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Available operations:\n\n" + describe_pipe_functions()
    )

    parser.add_argument('-q', '--quiet', help="Don't report progress")
    parser.add_argument('-s', '--segments', default=True,
                        help="Only process segments")
    parser.add_argument('-r', '--regex', default=r'.*',
                        help="Regular expression to select files; defaults to '.*'")
    parser.add_argument('-e', '--exclude', default=None,
                        help="Regular expression for files to exclude")
    parser.add_argument('-i', '--input', default=None,
                        help="Input file of commands; defaults to script.txt")
    parser.add_argument('-o', '--output', default=os.getcwd(),
                        help="top directory for results")
    parser.add_argument('-d', '--delete', action='store_true',
                        help="Delete existing files before the run")
    parser.add_argument('--dry', action='store_true',
                        help='List the files that would be handled and where the output would be written')

    args = parser.parse_args()

    # First look for a valid path expressed in the input argument
    infile = ""
    if args.input:
        if os.path.isfile(args.input):
            infile = os.path.abspath(args.input)
    else:
        infile = 'script.txt'

    if args.output:
        os.chdir(args.output)

    # Look for marching orders in an file called script.txt
    assert os.path.isfile(infile), f"You must supply file script.txt in the output directory"
    order_text = open(infile, 'r').readlines()
    orders = []
    for line in order_text:
        # first remove any white space around equal signs
        line = re.sub(r' *= *', "=", line)
        fields = line.strip().split()
        routine = fields.pop(0).strip(',')
        if routine:
            func = globals()[routine]
            kwargs = {}
            for x in fields:
                k, v = x.split('=')
                kwargs[k] = decode_arg(v)
            orders.append((func, kwargs))

    if args.delete:
        # remove all contents of subdirectories first
        for candidate in os.listdir('.'):
            if os.path.isdir(candidate):
                shutil.rmtree(candidate, ignore_errors=True)

    include = re.compile(args.regex)
    exclude = re.compile(args.exclude) if args.exclude else None

    if args.dry:
        print(f"Dry run, storing output in base directory {os.getcwd()}")

    for file in DigFile.all_dig_files():

        if not include.search(file):
            continue
        if exclude and exclude.search(file):
            continue
        path = os.path.join(DigFile.dig_dir(), file)
        if args.segments:
            df = DigFile(path)
            if not df.is_segment:
                continue
        if args.dry:
            print(path)
        else:
            pipe = PNSPipe(path, orders)

    # restore the working directory (is this necessary?)
    os.chdir(curdir)
