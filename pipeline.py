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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from ProcessingAlgorithms.preprocess.digfile import DigFile


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
        make_spectrogram(self, **kwargs)
        self.baselines = []
        self.gaps = []
        self.jumpoff = None
        self.probe_destruction = None
        self.signals = []

        # How do we specify spectrogram parameters?

        for order in orders:
            routine, kwargs = order
            routine(self, **kwargs)

    def log(self, x):
        self.logfile.write(x + '\n')

    def __del__(self):
        try:
            # print("I'm in the destructor")
            self.logfile.flush()
            self.logfile.close()
        except:
            pass


def make_spectrogram(pipeline, **kwargs):
    from spectrogram import Spectrogram
    pipeline.log(f"make_spectrogram for {kwargs}")

    defaults = dict(
        t_start = kwargs.get('t_start'),
        ending = kwargs.get('ending'),
        wavelength = kwargs.get('wavelength', 1.55e-6),
        points_per_spectrum = kwargs.get('points_per_spectrum', 8192),
        overlap = kwargs.get('overlap', 0.5),
        window_function = kwargs.get('window_function'),
        form = kwargs.get('form', 'power'),
    )
    t0 = time()
    allargs = {**defaults, **kwargs}
    pipeline.spectrogram = Spectrogram(
        pipeline.df,
        **allargs)
    pipeline.log(f"  took {(time()-t0):.1f} s")


def find_baselines(pipeline, **kwargs):
    t0 = time()
    from baselines import baselines_by_squash as bline
    peaks, widths, heights = bline(pipeline.spectrogram)
    baseline_limit = kwargs.get('baseline_limit', 0.1)
    pipeline.baselines = peaks[heights > baseline_limit]
    dt = time() - t0
    pipeline.log(f"Baselines > {baseline_limit*100}%: {pipeline.baselines}")
    pipeline.log(f"  in {dt*1000:.2f} ms")


def find_signal(pipeline, **kwargs):
    """
    Start at t_start and look for a peak above the baseline
    """
    from scipy.signal import find_peaks

    ts = kwargs.get('t_start', 2e-5)  # guess 20 µs if no info provided
    sg = pipeline.spectrogram
    t_index = sg._time_to_index(ts)
    # average over a small time neighborhood
    spectrum = np.sum(sg.intensity[:, t_index - 2:t_index + 2], axis = 1)
    peaks, properties = find_peaks(spectrum, height = 0.001 * spectrum.max(),
                                   distance = 40)
    heights = properties['peak_heights']
    # produce an ordering of the peaks from high to low
    ordering = np.flip(np.argsort(heights))
    peak_index = peaks[ordering]
    peaks = sg.velocity[peak_index]
    hts = heights[ordering]
    hts = hts / hts[0]  # normalize to largest peak of 1 (e.g., the baseline)
    blines = sorted(pipeline.baselines)
    # filter out any peaks on or below the baseline
    hts = hts[peaks > blines[0]]
    peaks = peaks[peaks > blines[0]]
    # Our guess for the signal is now at the first peak

    pipeline.signal_guess = (ts, peaks[0])
    pipeline.log(f"find_signal guesses signal is at {pipeline.signal_guess}")


def follow_signal(pipeline, **kwargs):
    """We'll first look left, then right.
    """
    t0 = time()
    from ProcessingAlgorithms.SignalExtraction.peak_follower import PeakFollower
    follower = PeakFollower(
        pipeline.spectrogram,
        pipeline.signal_guess,
        direction = -1)

    follower.run()
    follower.reverse()
    follower.run()

    signal = pd.DataFrame(follower.results)
    format_dict = dict(
        times = lambda x: f"{x*1e6:.3f}",
        velocities = lambda x: f"{x:.1f}",
        intensities = lambda x: f"{x:.1f}"
    )
    signal.style.format(format_dict)
    pipeline.signals.append(signal)
    dt = time() - t0
    pipeline.log(f"follow_signal generated {len(signal)} points in {dt:.2} s")
    pipeline.log(signal.to_string(
        formatters = format_dict, sparsify = False))

    plt.clf()
    plt.plot(signal['times'] * 1e6, signal['velocities'])
    plt.xlabel('$t~(\mu \mathrm{s})$')
    plt.ylabel('$v~(\mathrm{m/s})$')
    plt.savefig(os.path.join(pipeline.output_dir, 'follower.pdf'))


def gaussian_fit(pipeline, **kwargs):
    """
    Run through signals (which are pandas DataFrames) and
    add columns for gaussian centers, widths, and amplitudes.
    """
    from ProcessingAlgorithms.Fitting.gaussian import Gaussian
    t0 = time()
    for signal in pipeline.signals:
        # First add the requisite columns
        blanks = np.zeros(len(signal)) + np.nan
        signal['center'] = blanks
        signal['width'] = blanks
        signal['amplitude'] = blanks

        for n in range(len(signal)):
            row = signal.iloc[n]
            t_index = row['time_index']
            vfrom, vto = row['velocity_index_spans']
            powers = pipeline.spectrogram.intensity[vfrom:vto, t_index]
            speeds = pipeline.spectrogram.velocity[vfrom:vto]
            gus = Gaussian(speeds, powers)
            if gus.valid:
                signal.loc[n, 'center'] = gus.center
                signal.loc[n, 'width'] = gus.width
                signal.loc[n, 'amplitude'] = gus.amplitude

        signal['dcenter'] = signal.center - signal.velocities
    dt = time() - t0
    pipeline.log(f"\ngaussian_fit took {dt:.2f} s\n")

    def oned(x): return f"{x:.1f}"

    format_dict = dict(
        times = lambda x: f"{x*1e6:.3f}",
        velocities = oned,
        intensities = oned,
        center = oned,
        width = oned,
        amplitude = oned,
        dcenter = lambda x: f"{x:.2f}"
    )
    for signal in pipeline.signals:
        pipeline.log(signal.to_string(
            formatters = format_dict, sparsify = False))
        pipeline.log("\n\n")
        plt.clf()
        plt.errorbar(signal['times'] * 1e6,
                     signal.dcenter, yerr = signal.width, fmt = 'b.',
                     markersize = 1.0, lw = 0.5)
        plt.xlabel(r'$t~(\mu \mathrm{s})$')
        plt.ylabel(r'$\delta v~(\mathrm{m/s})$')
        plt.savefig(os.path.join(pipeline.output_dir, 'gauss.pdf'))

    # def find_gaps(pipeline, *args, **kwargs):
    # """
    # Look for regions where the signal is constant at one extreme or the other.
    # """

    # def find_probe_destruction(self):
    # """
    # Look for signs of probe destruction
    # """


sample_orders = (
    (find_baselines, {}),
    (find_signal, dict(t_start = 2e-5)),
    (follow_signal, dict()),
    (gaussian_fit, dict())
)

if __name__ == '__main__':
    # sys.path.insert(0, '../')
    for seg in range(16):
        print(seg)
        pipe = Pipeline(
            f'../dig/new/CH_1_009/seg{seg:02d}.dig', sample_orders, overlap = 0.75)
    print("Done!")
