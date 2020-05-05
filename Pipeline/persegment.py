#!/usr/bin/env python3
# coding:utf-8

"""
  Author:  LANL Clinic 2019 --<lanl19@cs.hmc.edu>
  Purpose: Process a set of .dig files
  Created: 04/27/20

  Routines to process results after all segments
"""

import numpy as np
import os
from pnspipe import PNSPipe
import matplotlib.pyplot as plt

from ProcessingAlgorithms.Fitting.fit import Exponential, DoubleExponential
from plotter import COLORMAPS
DEFMAP = '3w_gby'


def make_spectrogram(pipe: PNSPipe, **kwargs):
    """
    Compute a spectrogram using the passed kwargs or
    built-in defaults and set the pipe.spectrogram field.
    Default values are:
      wavelength = 1.55e-6
      points_per_spectrum = 4096
      overlap = 7/8
      window_function = None
      form = 'power'
    """
    from spectrogram import Spectrogram

    defaults = dict(
        t_start=kwargs.get('t_start'),
        ending=kwargs.get('ending'),
        wavelength=kwargs.get('wavelength', 1.55e-6),
        points_per_spectrum=kwargs.get('points_per_spectrum', 4096),
        overlap=kwargs.get('overlap', 7 / 8),
        window_function=kwargs.get('window_function'),
        form=kwargs.get('form', 'power'),
    )
    allargs = dict({**defaults, **kwargs})
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
        pipe.log(f"probe destruction at {t_blast*1e6:.2f} µs", True)


def find_baselines(pipe: PNSPipe, **kwargs):
    """
    Compute baselines for pipe.spectrogram using the
    baselines_by_squash method. If baseline_limit is
    passed as a keyword argument, only keep baseline values
    that are larger than this value (which should be between
    0 and 1).
    """
    from ProcessingAlgorithms.SignalExtraction.baselines import baselines_by_squash as bline
    peaks, widths, heights = bline(pipe.spectrogram)
    baseline_limit = kwargs.get('baseline_limit', 0.01)
    pipe.baselines = peaks[heights > baseline_limit]
    blines = ", ".join([f"{x:.1f}" for x in pipe.baselines])
    pipe.log(f"Baselines > {baseline_limit*100}%: {blines}", True)


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


def analyze_noise(pipe: PNSPipe, **kwargs):
    """
    Attempt to analyze the noise statistics in the spectrogram.

    Possible kwargs:
        v_min:    defaults to bottom baseline
        v_max:    defaults to 5000
        t_min:    defaults to t_start
        t_max:    defaults to probe destruction
        n_bins:   defaults to 100
    """

    f = pipe.spectrogram.analyze_noise(**kwargs)

    pipe.log(f" + v_max = {f.v_max} = {100*f.v_max/pipe.spectrogram.v_max:.1f}%")
    pipe.log(f" + t_range = {f.t_min*1e6:0.1f} - {f.t_max*1e6:0.1f} µs")
    pipe.log(f" + n_bins = {f.n_bins}")
    pipe.log(f"mean = {f.mean:0.3f}, sigma = {f.stdev:0.3f}", True)
    try:
        f.plot(
            logy=True,
            legend=(0, 0),
            figsize=(8, 8),
            xlabel="Power",
            ylabel="Counts",
            title=pipe.segment_name
        )
        figname = os.path.join(pipe.segment_parent, f'Noise_{pipe.segment_name}.pdf')
        f.fig.savefig(figname)
        plt.close(f.fig)
        del f.fig
        if isinstance(f, DoubleExponential):
            pipe.results['noise'] = dict(
                figname=figname,
                beta=f.params[0],
                lam1=f.params[1],
                lam2=f.params[2],
                amp=f.params[3],
                mean=f.mean,
                stdev=f.stdev,
                chisq=f.chisq,
                prob=f.prob_greater
            )
        elif isinstance(f, Exponential):
            pipe.results['noise'] = dict(
                figname=figname,
                lamb=f.params[0],
                amp=f.params[1],
                mean=f.mean,
                chisq=f.chisq,
                prob=f.prob_greater
            )
    except Exception as eeps:
        print(eeps)
        pass


def follow_signal(pipe: PNSPipe, **kwargs):
    """
    Using the peak_follower, look first look left, then right.
    Optional kwargs:
      plot = extension of the plot file type to save, or None;
             default is 'pdf'
      image = (boolean) produce a spectrogram image with superimposed follower
      gaussian = 'start:stop:step' in microseconds to plot gaussian fits
    """
    from ProcessingAlgorithms.SignalExtraction.peak_follower import PeakFollower
    if not pipe.signal_guess:
        pipe.log("No signal_guess so not following a signal.")
        return

    plot = kwargs.get('plot', 'pdf')
    image = kwargs.get('image', False)
    gaussian = kwargs.get('gaussian')
    sg = pipe.spectrogram

    follower = PeakFollower(
        sg,
        pipe.signal_guess,
        direction=-1)
    pipe.followers.append(follower)

    follower.run()
    follower.reverse()
    follower.run()

    signal = follower.full_frame(pipe.results['noise']['mean'])
    signal.style.format(pipe.pandas_format)
    pipe.signals.append(signal)

    pipe.log(f"follow_signal generated {len(signal)} points")
    pipe.log(signal.to_string(
        formatters=pipe.pandas_format, sparsify=False))

    if plot:
        plt.clf()
        fig, axes = plt.subplots(
            nrows=3,
            ncols=1,
            sharex=True,
            squeeze=True,
            gridspec_kw=dict(
                height_ratios=[0.25, 0.25, 0.5],
                left=0.125,
                right=0.975,
                wspace=0.05
            )
        )
        intensity, uncertainty, main = axes
        t = signal['time'] * 1e6  # convert to microseconds

        main.plot(t, signal['peak_v'], 'ro-', markersize=1, linewidth=0.5)
        main.set_xlabel(r'$t$ ($\mu$s)')
        main.set_ylabel(r'$v$ (m/s)')
        uncertainty.errorbar(t, t * 0, yerr=signal.m_width, fmt='o',
                             elinewidth=0.5, markersize=0.25)
        uncertainty.set_ylabel(r'$\sigma$ (m/s)')

        intensity.semilogy(t, signal['peak_int'],
                           'go-', markersize=1, linewidth=0.25)
        intensity.set_ylabel('$I$')
        miny = np.floor(np.log10(np.min(signal['peak_int'])))
        maxy = np.ceil(np.log10(np.max(signal['peak_int'])))
        intensity.set_ylim(10**miny, 10**maxy)
        intensity.set_title(pipe.df.title, usetex=False)
        fig.savefig(os.path.join(pipe.segment_parent, f'fol-{pipe.segment_name}.{plot}'))
        plt.close(fig)

    if image:
        # Produce a plot that superimposes the follower on the
        # spectrogram, which should be clipped below by the baseline
        # and to the right by probe_destruction, if it exists
        t_range = (
            sg.time[0], pipe.probe_destruction if pipe.probe_destruction else sg.time[-1])
        r = follower.results
        try:
            v_min = pipe.baselines[0]
        except:
            v_min = 0
        if r['peak_v']:
            top = np.max(r['peak_v']) + 400.0
        else:
            top = 7000  # this is terrible!
        try:
            v_min = pipe.baselines[0]
        except:
            v_min = 0.0

        v_range = v_min, top
        time, velocity, intensity = sg.slice(t_range, v_range)
        max_intensity = np.max(intensity)
        image = plt.pcolormesh(time * 1e6, velocity,
                               np.log10(intensity + 0.0001 * max_intensity) * 10)
        image.set_cmap(COLORMAPS[DEFMAP])
        fig = plt.gcf()
        # attempt to set the lower boundary of the colormap
        try:
            noise = pipe.results['noise']['mean']
            image.set_clim(vmin=2 * noise)
        except:
            pass
        ax = plt.gca()
        ax.plot(signal['time'] * 1e6,
                signal['peak_v'], 'o', color="#FF8888", alpha=0.05, markersize=1)
        plt.colorbar(image, ax=ax, fraction=0.08)
        plt.xlabel(r'$t$ ($\mu$s)')
        plt.ylabel('$v$ (m/s)')
        plt.title(pipe.df.title, usetex=False)
        fig.savefig(os.path.join(pipe.segment_parent, f"sg-{pipe.segment_name}.jpg"))
        plt.close(fig)

    if gaussian:
        start, end, step = [1e-6 * float(x) for x in gaussian.split(':')]
        times = np.arange(start, end + 0.99 * step, step)
        for t in times:
            pretitle = [
                pipe.title,
                "$N=%d$" % (sg.points_per_spectrum),
            ]
            if sg.nfft:
                pretitle.append(f"padding = {int(sg.nfft/sg.points_per_spectrum)}")
            try:
                hood = follower.hood(t=t, expand=(2, 1))
                ut = 1e6 * hood.time
                timing = f", {ut:.3f}" + r" $\mu$s"
                hood.gaussian.plot(
                    xlabel="$v$ (m/s)",
                    ylabel="Intensity",
                    legend=(0, 1),
                    title=", ".join(pretitle) + timing
                )
                fig = plt.gcf()
                fig.set_size_inches(6, 4.5)
                fname = f"gauss_{int(ut*1000)}.pdf"
                fig.savefig(os.path.join(pipe.output_dir, fname))
                plt.close(fig)
            except Exception as eeps:
                print(f"Error {eeps} making gaussian for time {1e6*t:.2} µs")


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

    # neighborhood = kwargs.get('neighborhood', 10)
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
            Crit3 = np.abs(rows.peak_v - rows.gaussian_center) / \
                rows.gaussian_width > sigmas
            peak_gauss_discrepancies = rows[Crit3].index.to_numpy()
            all_discrepancies.union(peak_gauss_discrepancies)
        if peak_moment:
            Crit4 = np.abs(rows.peak_v -
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


# for name, obj in inspect.getmembers(sys.modules[__name__]):
#    register_pipe_function(obj)
