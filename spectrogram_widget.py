#!/usr/bin/env python3
# coding:utf-8
"""
  Author:  LANL 2019 clinic --<lanl19@cs.hmc.edu>
  Purpose: To represent a spectrogram in a Jupyter notebook
  with convenient controls
  Created: 09/26/19
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import ipywidgets as widgets
from IPython.display import display

from digfile import DigFile
from spectrogram import Spectrogram
from spectrum import Spectrum
from plotter import COLORMAPS

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
    """

    def __init__(self, digfile, **kwargs):
        """
        For the widget to behave in a Jupyter notebook, place
        %matplotlib widget
        at the top of the notebook. This requires that the package
        ipympl is installed, which can be done either with pip3
        or conda install ipympl.

        I also recommend editing ~/.jupyter/custom/custom.css to
        modify the definition of .container

        .container {
            width: 100% !important;
            margin-right: 40px;
            margin-left: 40px;
            }
        """
        if isinstance(digfile, str):
            self.digfile = DigFile(digfile)
            # sg = self.spectrogram = Spectrogram(spectrogram)
        else:
            assert isinstance(
                digfile, DigFile), "You must pass in a DigFile"
            self.digfile = digfile

        self.title = self.digfile.filename.split('/')[-1].replace("_", "\\_")
        # handle the keyword arguments here

        # Compute the base spectrogram (do we really need this?)
        self.spectrogram = Spectrogram(self.digfile, None, None)

        # Create the figure to display this spectrogram
        # It would be nice to make this bigger!
        gspec = {
            'width_ratios': [1.5, 6],
            'height_ratios': [5, ],
        }
        self.fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True,
                                      squeeze=True, gridspec_kw=gspec)
        self.axSpectrum, self.axSpectrogram = axes

        # At the moment, clicking on the image updates the spectrum
        # shown on the left axes. It would be nice to be more sophisticated and
        # allow for more sophisticated interactions, including the ability
        # to display more than one spectrum.
        self.fig.canvas.mpl_connect(
            'button_press_event', lambda x: self.handle_click(x))
        self.spectrum(None)

        self.image = None     # we will set in update_spectrogram
        self.colorbar = None  # we will set this on updating, based on the

        self.individual_controls = dict()
        self.controls = None
        self.make_controls()

        # create the call-back functions, and then display the controls

        display(self.controls)
        self.update_spectrogram()

    def make_controls(self):
        """
        Create the controls for this widget and store them in self.controls.
        """
        cd = self.individual_controls  # the dictionary of controls
        df = self.digfile
        cd['t_range'] = slide = ValueSlider(
            "Time (µs)", (0, 25), (df.t0, df.t_final), 1e6,
            readout_format=".1f"
        )
        slide.continuous_update = False
        slide.observe(
            lambda x: self.do_update(x), names="value")

        cd['velocity_range'] = slide = ValueSlider(
            "Velocity (km/s)",
            (0, 50),
            (0.0, self.spectrogram.v_max), 1e-3,
            readout_format=".1f",
            continuous_update=False
        )
        slide.observe(lambda x: self.update_velocity_range(),
                      names="value")

        imax = self.spectrogram.intensity.max()
        imin = imax - 200  # ??
        cd['intensity_range'] = slide = ValueSlider(
            "Color",
            (40, 70),
            (imin, imax),
            multiplier=1,
            readout_format=".0f",
            continuous_update=False
        )
        slide.observe(lambda x: self.update_color_range(),
                      names="value")

        cd['color_map'] = widgets.Dropdown(
            options=sorted(COLORMAPS.keys()),
            value='3w_gby',
            description='Color Map',
            disabled=False,
        )
        cd['color_map'].observe(lambda x: self.update_cmap(), names="value")

        cd['raw_signal'] = widgets.Checkbox(
            value=False,
            description="Show V(t)")
        cd['raw_signal'].observe(
            lambda b: self.show_raw_signal(b), names="value")

        cd['spectrum_size'] = slide = widgets.IntSlider(
            value=13, min=8, max=18, step=1,
            description="FFT 2^n"
        )
        slide.continuous_update = False
        slide.observe(lambda x: self.overhaul(points_per_spectrogram=2 ** x['new']),
                      names="value")

        cd['shift'] = slide = widgets.IntSlider(
            description='Shift',
            value=self.spectrogram.points_per_spectrum - self.spectrogram.shift,
            min=1,
            max=2 * self.spectrogram.points_per_spectrum,
            step=1)
        slide.continuous_update = False
        slide.observe(lambda x: self.overhaul(
            shift=self.spectrogram.points_per_spectrum - x['new']),
            names="value")

        self.controls = widgets.HBox([
            widgets.VBox([
                cd['t_range'], cd['velocity_range'],
                cd['intensity_range'], cd['spectrum_size']
            ]),
            widgets.VBox([
                cd['color_map'],
                cd['raw_signal'],
                cd['shift']
            ])
        ])

    def range(self, var):
        if var in self.individual_controls:
            return self.individual_controls[var].range
        return None

    def do_update(self, what):
        self.update_spectrogram()

    def show_raw_signal(self, box):
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
            self.axSpectrogram.plot(tvals, vvals,
                                    'r-', alpha=0.5)
        else:
            try:
                del self.axSpectrogram.lines[0]
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
            except:
                pass

    def overhaul(self, **kwargs):
        """
        A parameter affecting the base spectrogram has been changed, so
        we need to recompute everything.
        """
        self.spectrogram.set(**kwargs)
        self.update_spectrogram()

    def update_spectrogram(self):
        """
        Recompute and display everything
        """
        sg = self.spectrogram
        trange = self.range('t_range')
        vrange = self.range('velocity_range')

        # extract the requisite portions
        times, velocities, intensities = sg.slice(trange, vrange)

        # Having recomputed the spectrum, we need to set the yrange

        # of the color map slider
        cmin = sg.intensity.min()
        cmax = sg.intensity.max()
        self.individual_controls['intensity_range'].range = (cmin, cmax)

        # if we have already displayed an image, remove it
        if self.colorbar:
            self.colorbar.remove()
            self.colorbar = None
        if self.image:
            self.image.remove()
            self.image = None

        self.image = self.axSpectrogram.pcolormesh(
            times * 1e6, velocities, intensities)

        self.colorbar = self.fig.colorbar(self.image, ax=self.axSpectrogram)

        self.axSpectrogram.set_title(self.title)
        self.axSpectrogram.set_xlabel('Time ($\mu$s)')
        self.axSpectrogram.set_xlim(* (np.array(trange) * 1e6))
        self.axSpectrum.set_ylabel('Velocity (m/s)')
        self.update_velocity_range()
        self.update_color_range()
        self.update_cmap()

    def update_cmap(self):
        mapname = self.individual_controls['color_map'].value
        self.image.set_cmap(COLORMAPS[mapname])

    def update_velocity_range(self):
        vmin, vmax = self.range('velocity_range')
        self.axSpectrogram.set_ylim(vmin, vmax)
        self.axSpectrum.set_ylim(vmin, vmax)

    def update_color_range(self):
        self.image.set_clim(self.range('intensity_range'))

    def handle_click(self, event):
        t, v = event.xdata * 1e-6, event.ydata
        # Compute a spectrum
        # we should do better about the length
        try:
            self.spectrum(t)
        except Exception as eeps:
            pass

    def spectrum(self, the_time):
        if the_time is None:
            # Initialize the axes
            self.axSpectrum.plot([0, 1], [0, 1], 'r-')
            self.axSpectrum.grid(axis='x', which='both', color='b', alpha=0.4)
        else:
            delta_t = self.spectrogram.points_per_spectrum / 2 * \
                self.digfile.dt
            the_spectrum = Spectrum(
                self.digfile.values(the_time - delta_t,
                                    the_time + delta_t),
                self.digfile.dt,
                remove_dc=True)
            line = self.axSpectrum.lines[0]
            intensities = the_spectrum.db
            line.set(xdata=intensities, ydata=the_spectrum.velocities)

            self.axSpectrum.set_xlim(
                (intensities.mean(), intensities.max()))
            # We should also add a line to the spectrogram showing where
            # the spectrum came from.
            if not self.axSpectrogram.lines:
                self.axSpectrogram.plot([0, 0], [0, 1], 'r-', alpha=0.33)
            line = self.axSpectrogram.lines[0] # this won't scale when we add baselines
            tval = the_time * 1e6 # convert to microseconds
            line.set(xdata=[tval, tval], ydata=[0, self.spectrogram.v_max])


if __name__ == '__main__':
    pass  # unittest.main()
