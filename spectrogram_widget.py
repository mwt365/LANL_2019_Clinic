#!/usr/bin/env python3
# coding:utf-8
"""
::

  Author:  LANL 2019 clinic --<lanl19@cs.hmc.edu>
  Purpose: To represent a spectrogram in a Jupyter notebook
  with convenient controls
  Created: 09/26/19
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import ipywidgets as widgets
from matplotlib import widgets as mwidgets
from IPython.display import display

from digfile import DigFile
from spectrogram import Spectrogram
from spectrum import Spectrum
from plotter import COLORMAPS
from gaussian import Gaussian
from gaussian_follow import GaussianFitter
from peak_follower import PeakFollower
from template_matcher import TemplateMatcher
from ImageProcessing.Templates.templates import opencv_start_pattern2
from matplotlib.patches import Rectangle
import time as Time


DEFMAP = '3w_gby'  # should really be in an .ini file


class ValueSlider(widgets.FloatRangeSlider):
    """
    A Slider to represent the range that we
    wish to display. The slider maps linearly over the range
    specified by yrange = (ymin, ymax). If a multiplier different
    from 1 is given, then values displayed by the widget
    will be divided by the multiplier to yield "real" values.
    So, for example, if "real" time values are in seconds, but
    we wish to display microseconds, the multiplier is 1e6 and
    values used internally get divided by 1e6 to yield true
    values for this parameter.
    Getters and setters
    are defined for ymin, ymax, and range.
    """

    def __init__(self, description,
                 initial_value,  # express as percentages (0, 50)
                 yrange,        # in true (not scaled) values
                 multiplier=1,
                 **kwargs):
        super().__init__(
            description=description,
            min=yrange[0] * multiplier,
            max=yrange[1] * multiplier,
            readout=True,
            value=[multiplier * (yrange[0] + 0.01 * x * (
                yrange[1] - yrange[0])) for x in initial_value],
            layout=widgets.Layout(width='400px'),
            **kwargs)
        self.multiplier = multiplier
        self._ymin = yrange[0] * multiplier
        self._ymax = yrange[1] * multiplier

    @property
    def ymin(self):
        return self._ymin

    @ymin.setter
    def ymin(self, v):
        self._ymin = v

    @property
    def ymax(self):
        return self._ymax

    @ymax.setter
    def ymax(self, v):
        self._ymax = v

    @property
    def range(self):
        return [v / self.multiplier for v in self.value]

    @range.setter
    def range(self, val):  # fix?
        assert isinstance(val, (list, tuple)) and len(val) == 2
        self._ymin, self._ymax = val


class PercentSlider(widgets.IntRangeSlider):
    """
    A Slider to represent the percentage of a range that we
    wish to display. The slider maps linearly over the range
    specified by yrange = (ymin, ymax). Getters and setters
    are defined for ymin, ymax, and range.
    """

    def __init__(self, description, initial_value, yrange):
        super().__init__(
            description=description,
            min=0, max=100, readout=True,
            value=initial_value,
            layout=widgets.Layout(width='400px')
        )
        self._ymin = yrange[0]
        self._ymax = yrange[1]

    @property
    def ymin(self):
        return self._ymin

    @ymin.setter
    def ymin(self, v):
        self._ymin = v

    @property
    def ymax(self):
        return self._ymax

    @ymax.setter
    def ymax(self, v):
        self._ymax = v

    @property
    def range(self):
        dy = 0.01 * (self.ymax - self.ymin)
        return [v * dy + self.ymin for v in self.value]

    @range.setter
    def range(self, val):
        assert isinstance(val, (list, tuple)) and len(val) == 2
        self._ymin, self._ymax = val


class SpectrogramWidget:
    """
    A Jupyter notebook widget to represent a spectrogram, along with
    numerous controls to adjust the appearance of the spectrogram.
    For the widget to behave in a Jupyter notebook, place::

        %matplotlib widget

    at the top of the notebook. This requires that the package
    ipympl is installed, which can be done either with pip3
    or conda install ipympl.

    I also recommend editing ~/.jupyter/custom/custom.css to
    modify the definition of .container::

        .container {
            width: 100% !important;
            margin-right: 40px;
            margin-left: 40px;
            }

    **Inputs**

    - digfile: either a string or DigFile
    - kwargs: optional keyword arguments. These are passed to
      the Spectrogram constructor and to the routine that
      creates the control widgets.

    **Data members**

    - digfile: the source data DigFile
    - title: the title displayed above the spectrogram
    - baselines: a list of baseline velocities
    - spectrogram: a Spectrogram object deriving from digfile
    - fig:
    - axSpectrum:
    - axSpectrogram:
    - image:
    - colorbar:
    - individual_controls: dictionary of widgets
    - controls:
    """
    _gspec = {
        'width_ratios': [6, 1.25],
        'height_ratios': [1],
        'wspace': 0.05,
        'left': 0.075,
        'right': 0.975,
    }

    def __init__(self, *args, **kwargs):
        """
        If one passes in a single unnamed arg, it can either be a digfile,
        a string pointing to a digfile, or a two-dimensional ndarray. 
        If we are founded on a dig file, it is possible to recompute
        things. Only a subset of operations are possible when we're
        based on a two-dimensional array, but perhaps that is sometimes
        desirable.
        """
        self.digfile = None
        if len(args) == 1:
            arg = args[0]
            if isinstance(arg, str):
                self.digfile = DigFile(arg)
            elif isinstance(arg, DigFile):
                self.digfile = arg
        if self.digfile == None and len(args):
            # Let's see if we have enough information to display a spectrogram
            # That means we have a two-dimensional ndarray, and possibly corresponding
            # time and velocity arrays. The signature would be (intensity, [times, velocities]).
            arg = args[0]
            if isinstance(arg, np.ndarray) and len(arg.shape) == 2:
                self._static = {'intensity': arg, }

                if len(args) == 3:
                    self._static['time'] = np.asarray(args[1])
                    self._static['velocity'] = np.asarray(args[2])
                else:
                    self._static['time'] = np.arange(0, -1 + arg.shape[1])
                    self._static['velocity'] = np.arange(
                        0, -1 + arg.shape[0])

            assert hasattr(
                self, '_static'), "Inappropriate arguments passed to the SpectrogramWidget constructor"

        # If LaTeX is enabled in matplotlib, underscores in the title
        # cause problems in displaying the histogram. However, we can
        # solve this by not using latex in displaying the title, so there
        # is no need to alter characters here.
        self.title = "" if self.static else self.digfile.filename.split(
            '/')[-1]
        self.baselines = []

        # Compute the base spectrogram (do we really need this?)
        if self.dig:
            self.spectrogram = Spectrogram(
                self.digfile, None, None, **kwargs)

        self.fig, axes = plt.subplots(
            nrows=1, ncols=2, sharey=True,
            squeeze=True, gridspec_kw=self._gspec)
        self.axSpectrogram, self.axSpectrum = axes

        self.subfig = None
        self.axTrack = None
        self.axSpare = None

        # At the moment, clicking on the image updates the spectrum
        # shown on the left axes. It would be nice to be more sophisticated and
        # allow for more sophisticated interactions, including the ability
        # to display more than one spectrum.
        self.fig.canvas.mpl_connect(
            'button_press_event', lambda x: self.handle_click(x))
        self.fig.canvas.mpl_connect(
            'key_press_event', lambda x: self.handle_key(x))

        self.spectrum(None, "")

        self.image = None     # we will set in update_spectrogram
        self.colorbar = None  # we will set this on updating, based on the

        self.peak_followers = []  # will hold any PeakFollowers
        self.spectra = []         # will hold spectra displayed at right
        self.spectra_in_db = True  # should spectra be displayed in db?

        self.controls = dict()    # widgets stored by name
        self.layout = None        # how the controls get laid out
        self.selecting = False    # we are not currently selecting a ROI
        self.roi = []             # and we have no regions of interest
        self.threshold = None
        self.make_controls(**kwargs)

        display(self.layout)
        self.update_spectrogram()

    @property
    def static(self):
        "If we are not associated with a dig file, return True"
        return hasattr(self, '_static')

    @property
    def dig(self):
        "True if we are associated with a dig file and can recompute the spectrogram"
        return isinstance(self.digfile, DigFile)

    @property
    def intensity(self):
        "Return the two-dimensional array of intensities"
        if self.dig:
            return self.spectrogram.intensity
        else:
            return self._static['intensity']

    def make_controls(self, **kwargs):
        """
        Create the controls for this widget and store them in self.controls.
        """
        cd = self.controls  # the dictionary of controls
        t_range = kwargs.get('t_range', (0, 100))

        # If we are associated with a dig file, include controls that
        # allow recomputation of the spectrogram
        if self.dig:
            df = self.digfile

            # FFT size  ###########################################
            # Set the size of each spectrum
            pps = kwargs.get('points_per_spectrum', 8192)
            val = int(np.log2(pps))
            cd['spectrum_size'] = slide = widgets.IntSlider(
                value=val, min=8, max=18, step=1,
                description="FFT 2^n"
            )
            slide.continuous_update = False
            slide.observe(lambda x: self.overhaul(
                points_per_spectrum=2 ** x['new']), names="value")

            # Set the overlap percentage of successive time intervals
            cd['overlap'] = slide = widgets.FloatSlider(
                description='Overlap %',
                value=100.0 * self.spectrogram.overlap,
                min=0,
                max=100)
            slide.continuous_update = False
            slide.observe(lambda x: self.overhaul(
                overlap=x['new'] * 0.01),
                names="value")

            # Thumbnail  ###########################################
            # Display a thumbnail of the raw signal
            cd['raw_signal'] = widgets.ToggleButton(
                value=False,
                description="Show V(t)",
                button_style='',
                icon='check'
            )
            cd['raw_signal'].observe(
                lambda b: self.show_raw_signal(b), names="value")

            # Time range ###########################################

            cd['t_range'] = slide = ValueSlider(
                "Time (µs)", t_range, (df.t0, df.t_final), 1e6,
                readout_format=".1f"
            )
            slide.continuous_update = False

            # Velocity range ###########################################
            cd['velocity_range'] = slide = ValueSlider(
                "Velocity (km/s)",
                (0, 50),
                (0.0, self.spectrogram.v_max), 1e-3,
                readout_format=".1f",
                continuous_update=False
            )
        else:
            d = self._static
            cd['t_range'] = ValueSlider(
                "Time", t_range, (d['time'][0], d['time'][-1]), 1e-3,
                readout_format=".1f",
                continuous_update=False
            )
            cd['velocity_range'] = ValueSlider(
                "Velocity (km/s)",
                (0, 100),
                (0.0, d['velocity'][-1]), 1e-3,
                readout_format=".1f",
                continuous_update=False
            )

        cd['t_range'].observe(
            lambda x: self.do_update(x), names="value")
        cd['velocity_range'].observe(lambda x: self.update_velocity_range(x),
                                     names="value")

        # Color range ###########################################

        imax = self.intensity.max()
        if self.dig:
            hl = self.spectrogram.histogram_levels
            imin = hl['tens'][3]
            # Let's figure out the range likely to produce a clear
            # image. Put the 50% point at the bottom end and the 95%
            # at the top?

            def scale(x): return int(100.0 * (x - imin) / (imax - imin))
            start_range = (scale(hl['tens'][8]), scale(hl['tenths'][9]))
        else:
            imin = imax - 150  # ?? this is bad and assumes db
            start_range = (40, 70)

        cd['intensity_range'] = slide = ValueSlider(
            "Color",
            start_range,
            (imin, imax),
            multiplier=1,
            readout_format=".0f",
            continuous_update=False
        )
        slide.observe(lambda x: self.update_color_range(),
                      names="value")

        # Threshold percentage #####################################
        cd['threshold'] = slide = widgets.FloatSlider(
            description='Noise floor %',
            value=0,
            min=0,
            max=100.0,
            continuous_update=False
        )
        slide.observe(lambda x: self.update_threshold(
            x['new']), names="value")

        # Color map selector ###########################################
        the_maps = sorted(COLORMAPS.keys())
        the_maps.append('Computed')
        cd['color_map'] = widgets.Dropdown(
            options=the_maps,
            value='3w_gby',
            description='Color Map',
            disabled=False,
        )
        cd['color_map'].observe(lambda x: self.update_cmap(),
                                names="value")

        # Click selector  ###########################################
        # What to do when registering a click in the spectrogram
        cd['clicker'] = widgets.Select(
            options=("Spectrum (dB)", "Spectrum", "Peak", "Gauss", "Template_Matching"),
            value='Spectrum (dB)',
            description="Click:",
            disabled=False
        )

        cd['marquee'] = mwidgets.RectangleSelector(
            self.axSpectrogram,
            lambda eclick, erelease: self.RSelect(eclick, erelease),
            interactive=True,
            useblit=True,
            rectprops=dict(facecolor='yellow', edgecolor='red',
                           alpha=0.2, fill=True),
            drawtype='box'
        )
        cd['marquee'].set_active(False)

        # Clear spectra ###########################################
        cd['clear_spectra'] = widgets.Button(
            description="Clear Spectra"
        )
        cd['clear_spectra'].on_click(lambda b: self.clear_spectra())

        # Clear peak_followers ###########################################
        cd['clear_followers'] = widgets.Button(
            description="Clear Peak Followers"
        )
        cd['clear_followers'].on_click(lambda b: self.clear_followers())

        # Computing baselines ###########################################
        cd['baselines'] = widgets.Dropdown(
            options=('_None_', 'Squash', 'FFT'),
            value='_None_',
            description='Baselines',
            disabled=False
        )
        cd['baselines'].observe(
            lambda x: self.update_baselines(x["new"]),
            names="value")

        columns = [
            'color_map;t_range;velocity_range;intensity_range;threshold',
            'clicker;clear_spectra;clear_followers',
        ]
        if self.dig:
            columns.append(
                'spectrum_size;overlap;raw_signal;baselines')

        vboxes = []
        for col in columns:
            vboxes.append(widgets.VBox([cd[x] for x in col.split(';')]))
        self.layout = widgets.HBox(vboxes)

    def range(self, var):
        "Return the range of the named control, or None if not found."
        if var in self.controls:
            return self.controls[var].range
        return None

    def RSelect(self, eclick, erelease):
        "Called when self.selecting is True and the marquee is active"
        if self.selecting:
            t0, t1 = eclick.xdata, erelease.xdata
            v0, v1 = eclick.ydata, erelease.ydata
            # make sure they are in the right order
            if t1 < t0:
                t0, t1 = t1, t0
            if v1 < v0:
                v0, v1 = v1, v0
            self.roi.append(dict(time=(t0, t1), velocity=(v0, v1)))

    def do_update(self, what):
        self.update_spectrogram()

    def show_raw_signal(self, box):
        """
        Display or remove the thumbnail of the time series data
        at the top of the spectrogram window.
        """
        if box.new:
            # display the thumbnail
            t_range = self.range('t_range')
            thumb = self.digfile.thumbnail(*t_range)
            # we have to superpose the thumbnail on the
            # existing velocity axis, so we need to rescale
            # the vertical.
            tvals = thumb['times'] * 1e6  # convert to µs
            yvals = thumb['peak_to_peak']
            ylims = self.axSpectrum.get_ylim()
            # Map the thumbnail to the top 20%
            ymax = yvals.max()
            yrange = ymax - yvals.min()
            yscale = 0.2 * (ylims[1] - ylims[0]) / yrange
            vvals = ylims[1] - yscale * (ymax - yvals)
            self.raw = self.axSpectrogram.plot(tvals, vvals,
                                               'r-', alpha=0.5)[0]
        else:
            try:
                self.axSpectrogram.lines.remove(self.raw)
                self.raw = None
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
            except:
                pass

    def overhaul(self, **kwargs):
        """
        A parameter affecting the base spectrogram has been changed, so
        we need to recompute everything.
        """
        if self.dig:
            self.spectrogram.set(**kwargs)
        self.update_spectrogram()

    def update_spectrogram(self):
        """
        Recompute and display everything
        """

        intense = self.spectrogram.intensity if self.dig else self._static['intensity']

        # Having recomputed the spectrum, we need to set the yrange

        # of the color map slider
        cmin = intense.min()
        cmax = intense.max()
        self.controls['intensity_range'].range = (cmin, cmax)
        self.display_spectrogram()

    def display_spectrogram(self):
        """

        """
        trange = self.range('t_range')
        vrange = self.range('velocity_range')

        if self.dig:
            # extract the requisite portions
            times, velocities, intensities = self.spectrogram.slice(
                trange, vrange)
        else:
            d = self._static
            times, velocities, intensities = d['time'], d['velocity'], d['intensity']

        # if we have already displayed an image, remove it
        if self.colorbar:
            self.colorbar.remove()
            self.colorbar = None
        if self.image:
            self.image.remove()
        self.image = None

        if self.threshold:
            intensities[intensities < self.threshold] = self.threshold

        self.image = self.axSpectrogram.pcolormesh(
            times * 1e6, velocities, intensities)

        self.colorbar = self.fig.colorbar(self.image, ax=self.axSpectrogram,
                                          fraction=0.08)

        self.axSpectrogram.set_title(self.title, usetex=False)
        self.axSpectrogram.set_xlabel('Time ($\mu$s)')
        self.axSpectrogram.set_xlim(* (np.array(trange) * 1e6))
        self.axSpectrogram.set_ylabel('Velocity (m/s)')
        self.update_velocity_range()
        self.update_color_range()
        self.update_cmap()

    def update_threshold(self, x):
        n = x
        sg = self.spectrogram
        if int(x) == 0:
            self.threshold = None
        else:
            if x < 90:
                threshold = sg.histogram_levels['tens'][x // 10]
            elif x < 99:
                threshold = sg.histogram_levels['ones'][int(x - 90)]
            else:
                threshold = sg.histogram_levels['tenths'][int(10 * (x - 99))]
            self.threshold = self.spectrogram.transform(threshold)
        self.display_spectrogram()

    def update_cmap(self):
        """
        Update the color map used to display the spectrogram
        """
        mapname = self.controls['color_map'].value
        if mapname == 'Computed':
            from generate_color_map import make_spectrogram_color_map
            mapinfo = make_spectrogram_color_map(
                self.spectrogram, 4, mapname)
            maprange = (mapinfo['centroids'][1], mapinfo['centroids'][-2])
            self.controls['intensity_range'].value = maprange
        self.image.set_cmap(COLORMAPS[mapname])

    def update_velocity_range(self, info=None):
        """
        Update the displayed velocity range using values obtained
        from the 'velocity_range' slider.
        """
        if info:
            old_vmin, old_vmax = info['old']
            vmin, vmax = info['new']
            if vmax > old_vmax or vmin < old_vmin:
                return self.update_spectrogram()
        vmin, vmax = self.range('velocity_range')
        self.axSpectrogram.set_ylim(vmin, vmax)
        self.axSpectrum.set_ylim(vmin, vmax)

    def update_color_range(self):
        self.image.set_clim(self.range('intensity_range'))

    def handle_click(self, event):
        try:
            # convert time to seconds
            t, v = event.xdata * 1e-6, event.ydata
        except:
            return 0
        if self.selecting:
            return 0
        # Look up what we should do with the click
        action = self.controls['clicker'].value
        try:
            if 'Spectrum' in action:
                self.spectrum(t, action)
            elif 'Template_Matching' in action:
                self.match_templates(t, v)
            else:
                self.follow(t, v, action)

        except Exception as eeps:
            pass

    def handle_key(self, event):
        try:
            # convert time to seconds
            t, v = event.xdata * 1e-6, event.ydata
        except:
            pass
        char = event.key
        if char == 'x':
            # remove the all spectra
            self.clear_spectra()
        if char in ('m', 'M'):
            self.selecting = not self.selecting
            self.controls['marquee'].set_active(self.selecting)
        if char in "0123456789":
            n = int(char)
            # self.fan_out(int(char))
            self.gauss_out(n)
        if char in ('a', 'A') and self.roi:
            self.analyze_roi()

    def clear_spectra(self):
        """Remove all spectra from axSpectrum and the corresponding
        markers from axSpectrogram
        """
        for x in self.spectra:
            self.axSpectrogram.lines.remove(x['marker'])
            self.axSpectrum.lines.remove(x['line'])
        self.spectra = []
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def clear_followers(self):
        """Remove all followers"""
        for x in self.peak_followers:
            self.axSpectrogram.lines.remove(x.line)
        self.peak_followers = []
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def follow(self, t, v, action):
        """Attempt to follow the path starting with the clicked
        point."""

        if action == "Gauss":
            fitter = GaussianFitter(self.spectrogram, (t, v))
            self.gauss = fitter
        elif action == "Peak":
            follower = PeakFollower(self.spectrogram, (t, v))
            # self.peak = follower
            self.peak_followers.append(follower)
            follower.run()
            tsec, v = follower.v_of_t
            follower.line = self.axSpectrogram.plot(
                tsec * 1e6, v, 'r-', alpha=0.4)[0]
        # print("Create a figure and axes, then call self.gauss.show_fit(axes)")

    def gauss_out(self, n: int):
        if n >= len(self.peak_followers):
            return 0
        pf = self.peak_followers[n]
        times, centers, widths, amps = [], [], [], []
        vind = pf.frame['velocity_index_spans'].to_numpy()
        tind = pf.frame['time_index'].to_numpy()
        sp = self.spectrogram
        for j in range(len(tind)):
            t = sp.time[tind[j]] * 1e6
            vfrom, vto = vind[j]
            powers = sp.power(sp.intensity[vfrom:vto, tind[j]])
            speeds = sp.velocity[vfrom:vto]

            gus = Gaussian(speeds, powers)
            if gus.valid:
                times.append(t)
                centers.append(gus.center)
                widths.append(gus.width)
                amps.append(gus.amplitude)
        fig, axes = plt.subplots(nrows=1, ncols=3, squeeze=True)
        ax1, ax2, ax3 = axes
        ax1.errorbar(times, centers, fmt='b-', yerr=widths)
        ax1.set_xlabel(r'$t$ ($\mu$s)')
        ax1.set_ylabel(r'$v$ (m/s)')

        ax2.plot(times, widths, 'r-')
        ax2.set_xlabel(r'$t$ ($\mu$s)')
        ax2.set_ylabel(r'$\delta v$ (m/s)')

        ax3.plot(times, amps, 'g-')
        ax3.set_xlabel(r'$t$ ($\mu$s)')
        ax3.set_ylabel('Amplitude')

        # Store the values for later access
        if not hasattr(self, "gauss_outs"):
            self.gauss_outs = [None for x in range(len(self.peak_followers))]
        else:
            while len(self.gauss_outs) < len(self.peak_followers):
                self.gauss_outs.append(None)
        self.gauss_outs[n] = dict(
            time=np.array(times),
            center=np.array(centers),
            width=np.array(widths),
            amplitude=np.array(amps)
        )

    def analyze_roi(self):
        """
        Extract the region(s) of interest and process them
        """
        for roi in self.roi:
            analyze_region(self.spectrogram, roi['time'])

    def fan_out(self, n: int):
        """Produce a zoomed in version of this trace, showing
        the neighborhood around the determined peak.
        """
        if n >= len(self.peak_followers):
            return 0
        pf = self.peak_followers[n]
        vind = pf.frame['velocity_index_spans'].to_numpy()
        tind = pf.frame['time_index'].to_numpy()
        self.subfig, axes = plt.subplots(
            nrows=1, ncols=2, sharey=True,
            squeeze=True, gridspec_kw=self._gspec)
        self.axSpare, self.axTrack = axes

        # We will create a "waterfall" of curves surrounding
        # the peaks, each offset by a bit. The x axis will
        # represent intensity, with subsequent time traces offset
        # by an amount I need to determine. The y axis
        # is velocity.

        spans = []
        vvec = self.spectrogram.velocity  # shortcut to velocity vector
        tvec = self.spectrogram.time
        ivec = self.spectrogram.intensity

        # pre-extract a bunch of one-dimensional curves
        # and be sure to convert to power
        for n in range(len(tind)):
            vfrom, vto = vind[n]
            spans.append({
                'v': vvec[vfrom:vto],
                'power': self.spectrogram.power(ivec[vfrom:vto, tind[n]]),
                't': tvec[tind[n]] * 1e6,
            })

        maxima = np.array([np.max(x['power']) for x in spans])
        maxpower = maxima.max()
        # Let's set the offset between times to be one tenth of
        # the maxpower
        offset = 0.025 * maxpower

        for n in reversed(list(range(len(spans)))):
            span = spans[n]
            self.axTrack.plot(
                span['power'] + n * offset,
                span['v'],
                'b-',
                alpha=0.33
            )
        self.axTrack.set_ylabel('$v$')

    def update_baselines(self, method):
        """
        Handle the baselines popup menu
        """
        from baselines import baselines_by_squash
        blines = []
        self.baselines = []  # remove any existing baselines
        if method == "Squash":
            peaks, sigs, heights = baselines_by_squash(self.spectrogram)
            for n in range(len(heights)):
                if heights[n] > 0.1:
                    blines.append(peaks[n])

        # Now show the baselines in blines or remove any
        # if blines is empty

        if not blines:
            for b in self.baselines:
                self.axSpectrum.lines.remove(b['line'])
            self.baselines = []  # remove them
        else:
            edges = (
                self.spectrogram.intensity.min(),
                self.spectrogram.intensity.max()
            )
            for v in blines:
                bline = self.axSpectrum.plot(
                    [edges[0], edges[1]],
                    [v, v],
                    'k-',
                    alpha=0.4
                )
                self.baselines.append(dict(v=v, line=bline))

    def baseline_intensity(self):
        self.update_baselines("Squash")
        figgy = plt.figure()
        ax = figgy.add_subplot(1, 1, 1)
        sg = self.spectrogram
        for bline in self.baselines:
            index = sg._velocity_to_index(bline['v'])
            if index > 2:
                ax.semilogy(sg.time * 1e6, sg.power(sg.intensity[index, :]))
        ax.set_xlabel(r"$t$ ($\mu$ s)")
        ax.set_ylabel(r"Power")

    def spectrum(self, the_time: float, form: str):
        """
        Display a spectrum in the left axes corresponding to the
        passed value of the_time (which is in seconds).
        """
        _colors = ["r", "g", "b", "y"]
        if the_time is None:
            # Initialize the axes
            # self.axSpectrum.plot([0, 1], [0, 1], 'r-')
            self.axSpectrum.grid(axis='x', which='both',
                                 color='b', alpha=0.4)
        else:
            delta_t = self.spectrogram.points_per_spectrum / 2 * \
                self.digfile.dt
            the_spectrum = Spectrum(
                self.digfile.values(the_time - delta_t,
                                    the_time + delta_t),
                self.digfile.dt,
                remove_dc=True)
            # compute the level of the 90th percentile
            spec = dict(spectrum=the_spectrum)
            vals = the_spectrum.db
            ordering = np.argsort(vals)
            if self.baselines:
                blines = [x['v'] for x in self.baselines]
                n = -1
                while the_spectrum.velocities[ordering[n]] in blines:
                    n -= 1
            else:
                n = -1
            spec['max'] = vals[ordering[n]]
            noise_floor = int(n - 0.1 * len(vals))
            spec['90'] = vals[ordering[noise_floor]]

            # We need to worry about the format of the spectrum
            db = ('dB' in form)
            field = 'db' if db else 'power'
            the_line = self.axSpectrum.plot(
                getattr(the_spectrum, field),
                the_spectrum.velocities,
                _colors[len(self.spectra)],
                alpha=0.33
            )
            spec['line'] = the_line[0]

            tval = the_time * 1e6  # convert to microseconds
            marker = self.axSpectrogram.plot(
                [tval, tval],
                [0, self.spectrogram.v_max],
                _colors[len(self.spectra)],
                alpha=0.33)
            spec['marker'] = marker[0]

            self.spectra.append(spec)

            if db != self.spectra_in_db:
                self.spectra_in_db = db  # switch our mode
                # and replot all the spectra
                for spec in self.spectra:
                    li = spec['line']
                    sp = spec['spectrum']
                    li.set(xdata=getattr(sp, field), ydata=sp.velocities)

            self.axSpectrum.set_xlabel("Power (dB)" if db else "Power")
            if db:
                # we should order the values and set a limit at something
                # like the strongest decile
                ninety = max([x['90'] for x in self.spectra])
                peak = max([x['max'] for x in self.spectra])
                self.axSpectrum.set_xlim(ninety, peak)
            return 0
            line = self.axSpectrum.lines[0]
            intensities = the_spectrum.db
            line.set(xdata=intensities, ydata=the_spectrum.velocities)

            # We should also add a line to the spectrogram showing where
            # the spectrum came from.
            if not self.axSpectrogram.lines:
                self.axSpectrogram.plot([0, 0], [0, 1], 'r-', alpha=0.33)
            # this won't scale when we add baselines
            line = self.axSpectrogram.lines[0]

            line.set(xdata=[tval, tval], ydata=[0, self.spectrogram.v_max])


    def match_templates(self, time, velocity):

        # print("start")
        start = Time.time()

        template = opencv_start_pattern2
        template_start_index = 5 #where should the actual start index be in the template?

        span = 80

        dv = self.spectrogram.velocity[3*span]
        dt = self.spectrogram.time[span] * 1e6

        t = time * 1e6

        time_offset = dt / 2
        velo_offset = dv / 2

        box_coords = (t-time_offset, velocity-velo_offset)
        new_click = (box_coords[0]*1e-6, velocity-velo_offset)

        start_time = self.spectrogram.time[0] * 1e6
        start_time_offset = start_time * -1

        matcher = TemplateMatcher(self.spectrogram, new_click, template, span=span)

        times, velos = matcher.main()

        times[:] = times[:] + start_time_offset

        # print("done with matching")

        # patch = Rectangle( (t-time_offset, velocity-velo_offset), dt, dv, fill=False, color='b', alpha=0.15)
        # self.axSpectrogram.add_patch(patch)

        # self.axSpectrogram.plot( times, velos, 'bo', markersize=1.5, alpha=0.6)

        # print(times, velos)

        max_follower = 0
        max_index = 0
        max_tsecs = 0
        max_v = 0

        for i in range(len(times)):

            t = times[i] * 1e-6
            v = velos[i]

            follower = PeakFollower(self.spectrogram, (t, v))

            self.peak_followers.append(follower)
            follow_sum = follower.run()

            tsec, v = follower.v_of_t

            if follow_sum > max_follower:
                max_follower = follow_sum
                max_index = i
                max_tsecs = tsec
                max_v = v

        # follower.line = self.axSpectrogram.plot(
        #     max_tsecs * 1e6, max_v, 'r-', alpha=0.3)[0]

        most_likely_time = times[max_index] + (self.spectrogram.time[template_start_index] * 1e6)

        most_likely_velo = velos[max_index]

        end = Time.time()

        print("finished in: ", round(end-start, 3), " seconds")

        # print("done with following")

        # print(most_likely_time, most_likely_velo)

        self.axSpectrogram.plot( most_likely_time, most_likely_velo, 'ro', markersize=2.5, alpha=1)



