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


from ProcessingAlgorithms.preprocess.digfile import DigFile


def all_pipeline_functions():
    funcs = []
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isfunction(obj):
            sig = list(inspect.signature(obj).parameters.keys())
            if sig and sig[0] == 'pipeline' and len(sig) == 2:
                funcs.append(obj)
    return funcs


def describe_pipeline_functions():
    """
    List all defined pipeline functions, with their docstrings.
    """
    thestrs = [f"{x.__name__}\n{x.__doc__}" for x in all_pipeline_functions()]
    return "\n\n".join(thestrs)


class Pipeline:
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
        os.makedirs(self.rel_dir, exist_ok = True)
        # Open a log file
        self.logfile = open(
            os.path.join(self.output_dir, 'info.txt'),
            'w', buffering = 1)
        now = datetime.datetime.now()
        self.log(f"{self.filename} log, {now.strftime('%a %d %b %Y at %H:%M:%S')}")
        self.spectrogram = None
        self.baselines = []
        self.gaps = []
        self.jumpoff = None
        self.probe_destruction = None
        self.signals = []
        self.pandas_format = dict(
            times = lambda x: f"{x*1e6:.3f}",
            velocities = self.onedigit,
            intensities = self.onedigit,
            center = self.onedigit,
            width = self.onedigit,
            amplitude = self.onedigit,
            dcenter = lambda x: f"{x:.2f}"
        )

        # How do we specify spectrogram parameters?
        # If the first command is not a specification for computing
        # a spectrogram, use the defaults

        if orders[0][0] != make_spectrogram:
            orders.insert(0, (make_spectrogram, {}))

        for order in orders:
            routine, kwargs = order
            self.start(routine, kwargs)
            print(f"{routine.__name__} ...", end="", flush = True)
            routine(self, **kwargs)
            print(self.end())

    def onedigit(self, val):
        return f"{val:.1f}"

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
        except:
            pass


def make_spectrogram(pipeline, **kwargs):
    """
    Compute a spectrogram using the passed kwargs or
    built-in defaults and set the pipeline.spectrogram field.
    Default values are:
      wavelength = 1.55e-6
      points_per_spectrum = 8192
      overlap = 0.5
      window_function = None
      form = 'power'
    """
    from spectrogram import Spectrogram

    defaults = dict(
        t_start = kwargs.get('t_start'),
        ending = kwargs.get('ending'),
        wavelength = kwargs.get('wavelength', 1.55e-6),
        points_per_spectrum = kwargs.get('points_per_spectrum', 8192),
        overlap = kwargs.get('overlap', 0.5),
        window_function = kwargs.get('window_function'),
        form = kwargs.get('form', 'power'),
    )
    allargs = {**defaults, **kwargs}
    # list the arguments we use in the log
    for k, v in allargs.items():
        pipeline.log(f"+ {k} = {v}")

    pipeline.spectrogram = Spectrogram(pipeline.df, **allargs)


def find_baselines(pipeline, **kwargs):
    """
    Compute baselines for pipeline.spectrogram using the
    baselines_by_squash method. If baseline_limit is
    passed as a keyword argument, only keep baseline values
    that are larger than this value (which should be between
    0 and 1).
    """
    from baselines import baselines_by_squash as bline
    peaks, widths, heights = bline(pipeline.spectrogram)
    baseline_limit = kwargs.get('baseline_limit', 0)
    pipeline.baselines = peaks[heights > baseline_limit]
    pipeline.log(f"Baselines > {baseline_limit*100}%: {pipeline.baselines}")


def find_signal(pipeline, **kwargs):
    """
    Start at t_start and look for a peak above the baseline
    """
    from ProcessingAlgorithms.SignalExtraction.peak_follower import signal_finder

    t_start = float(kwargs.get('t_start', 2e-5))
    blines = sorted(pipeline.baselines)
    sg = pipeline.spectrogram
    guess = signal_finder(sg, blines[0] + 5 * sg.dv, t_start)
    pipeline.signal_guess = (t_start, guess)
    pipeline.log(f"find_signal guesses signal is at {pipeline.signal_guess}")


def find_signal_old(pipeline, **kwargs):
    """
    Start at t_start and look for a peak above the baseline
    """
    from scipy.signal import find_peaks

    # guess 20 µs if no info provided
    ts = float(kwargs.get('t_start', 2e-5))
    sg = pipeline.spectrogram
    t_index = sg._time_to_index(ts)
    # average over a small time neighborhood
    spectrum = np.sum(sg.intensity[:, t_index - 2:t_index + 2], axis = 1)
    peaks, properties = find_peaks(spectrum, height = 0.001 * spectrum.max(),
                                   distance = 40)
    try:
        heights = properties['peak_heights']
        # produce an ordering of the peaks from high to low
        ordering = np.flip(np.argsort(heights))
        peak_index = peaks[ordering]
        peaks = sg.velocity[peak_index]
        hts = heights[ordering]
        # normalize to largest peak of 1 (e.g., the baseline)
        hts = hts / hts[0]
        blines = sorted(pipeline.baselines)
        # filter out any peaks on or below the baseline
        # We'll slide up a couple of velocity steps, just to be sure
        main_baseline = blines[0] + 5 * sg.dv
        hts = hts[peaks > main_baseline]
        peaks = peaks[peaks > main_baseline]
        # Our guess for the signal is now at the first peak

        pipeline.signal_guess = (ts, peaks[0])
        pipeline.log(f"find_signal guesses signal is at {pipeline.signal_guess}")
    except:
        pipeline.signal_guess = None
        pipeline.log("find_signal could not find a signal")


def follow_signal(pipeline, **kwargs):
    """
    Using the peak_follower, look first look left, then right.
    Optional kwargs:
      plot = extension of the plot file type to save, or None;
             default is 'pdf'

    """
    from ProcessingAlgorithms.SignalExtraction.peak_follower import PeakFollower
    if not pipeline.signal_guess:
        pipeline.log("No signal_guess so not following a signal.")
        return

    plot = kwargs.get('plot', 'pdf')

    follower = PeakFollower(
        pipeline.spectrogram,
        pipeline.signal_guess,
        direction = -1)

    follower.run()
    follower.reverse()
    follower.run()

    signal = pd.DataFrame(follower.results)
    signal.style.format(pipeline.pandas_format)
    pipeline.signals.append(signal)

    pipeline.log(f"follow_signal generated {len(signal)} points")
    pipeline.log(signal.to_string(
        formatters = pipeline.pandas_format, sparsify = False))

    if plot:
        plt.clf()
        plt.plot(signal['times'] * 1e6, signal['velocities'])
        plt.xlabel('$t~(\mu \mathrm{s})$')
        plt.ylabel('$v~(\mathrm{m/s})$')
        plt.title(pipeline.df.basename, usetex=False)
        plt.savefig(os.path.join(pipeline.output_dir, 'follower.' + plot))


def gaussian_fit(pipeline, **kwargs):
    """
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
    from ProcessingAlgorithms.Fitting.gaussian import Gaussian
    from ProcessingAlgorithms.Fitting.moments import moment

    neighborhood = kwargs.get('neighborhood', 10)
    sigmas = kwargs.get('sigmas', 1.5)  # discrepancy greater than this
    # will be reported
    sg = pipeline.spectrogram

    for signal in pipeline.signals:
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
            vpeak = sg._velocity_to_index(row['velocities'])
            # vfrom, vto = row['velocity_index_span']
            vfrom, vto = vpeak - neighborhood, vpeak + neighborhood
            powers = sg.intensity[vfrom:vto, t_index]
            speeds = sg.velocity[vfrom:vto]
            mom = moment(speeds, powers)
            signal.loc[n, 'mean'] = mom['center']
            signal.loc[n, 'sigma'] = mom['std_dev']
            gus = Gaussian(
                speeds, powers,
                center = row['velocities'],
                width = sg.velocity[2] - sg.velocity[0]
            )
            if gus.valid:
                signal.loc[n, 'center'] = gus.center
                signal.loc[n, 'width'] = gus.width
                signal.loc[n, 'amplitude'] = gus.amplitude
                diff = gus.center - signal.loc[n, 'velocities']
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

                    plt.plot([row['velocities'], ],
                             [row['intensities'], ], 'k*')
                    plt.plot([row['mean'] + n * row['sigma'] for n in (-1, 0, 1)],
                             [row['intensities'] for x in (-1, 0, 1)],
                             'gs')
                    plt.plot(speeds, powers, 'r.')
                    vels = np.linspace(gus.center - 6 * gus.width,
                                       gus.center + 6 * gus.width, 100)
                    plt.plot(vels, gus(vels), 'b-', alpha = 0.5)
                    tval = f"${pipeline.spectrogram.time[t_index]*1e6:.2f}"
                    plt.title(tval + "$~ µs")
                    plt.xlabel("Velocity (m/s)")
                    plt.ylabel(r"Intensity")
                    plt.xlim(vmin, vmax)
                    plt.savefig(os.path.join(pipeline.output_dir, f'bad{t_index}.pdf'))
                    plt.close()
                    with open(os.path.join(pipeline.output_dir, f'bad{t_index}.txt'), 'w') as f:
                        f.write(pd.DataFrame(
                            {'power': powers, }, index = speeds).to_csv())

    for signal in pipeline.signals:
        pipeline.log(signal.to_string(
            formatters = pipeline.pandas_format, sparsify = False))
        pipeline.log("\n\n")
        plt.clf()
        fig, axes = plt.subplots(3, 1, sharex = True, figsize = (6, 6))
        top, middle, bottom = axes
        top.semilogy(signal.times * 1e6, signal.intensities)
        top.set_ylabel('Intensity')
        top.set_title(pipeline.df.basename)
        middle.plot(signal.times * 1e6, signal.velocities)
        middle.set_ylabel(r'$v~(\mathrm{m/s})$')
        bottom.errorbar(signal['times'] * 1e6,
                        signal.dcenter, yerr = signal.width, fmt = 'b.',
                        markersize = 1.0, lw = 0.5)
        bottom.set_xlabel(r'$t~(\mu \mathrm{s})$')
        bottom.set_ylabel(r'$\delta v~(\mathrm{m/s})$')
        plt.savefig(os.path.join(pipeline.output_dir, 'gauss.pdf'))


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
        description = 'Run a pipeline to process .dig files',
        prog = "pipeline",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog = "Available operations:\n\n" + describe_pipeline_functions()
    )

    parser.add_argument('-q', '--quiet', help = "Don't report progress")
    parser.add_argument('-s', '--segments', default = True,
                        help = "Only process segments")
    parser.add_argument('-r', '--regex', default = r'.*',
                        help = "Regular expression to select files; defaults to '.*'")
    parser.add_argument('-e', '--exclude', default = None,
                        help = "Regular expression for files to exclude")
    parser.add_argument('-i', '--input', default = None,
                        help = "Input file of commands; defaults to script.txt")
    parser.add_argument('-o', '--output', default = os.getcwd(),
                        help = "top directory for results")
    parser.add_argument('-d', '--delete', action = 'store_true',
                        help = "Delete existing files before the run")
    parser.add_argument('--dry', action = 'store_true',
                        help = 'List the files that would be handled and where the output would be written')

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
                shutil.rmtree(candidate, ignore_errors = True)

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
            pipe = Pipeline(path, orders)

    # restore the working directory (is this necessary?)
    os.chdir(curdir)
