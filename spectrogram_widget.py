#!/usr/bin/env python3
# coding:utf-8
"""
  Author:  LANL 2019 clinic --<lanl19@cs.hmc.edu>
  Purpose: To represent a spectrogram in a Jupyter notebook with convenient controls
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

        I also recommend editing ~/.jupyter/custom/custom.css to modify
        the definition of .container

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
        cd['t_range'] = PercentSlider(
            "Time Range", (0, 25), (df.t0, df.t_final))
        cd['t_range'].continuous_update = False
        cd['t_range'].observe(
            lambda x: self.do_update(x), names="value")

        cd['velocity_range'] = PercentSlider(
            "Velocity Range",
            (0, 50),
            (0.0, self.spectrogram.v_max)
        )
        cd['velocity_range'].continuous_update = False
        cd['velocity_range'].observe(lambda x: self.update_velocity_range(),
                                     names="value")

        cd['intensity_range'] = PercentSlider(
            "Color Range",
            (40, 70),
            (0, 1)  # really needs to get updated!
        )
        cd['intensity_range'].continuous_update = False
        cd['intensity_range'].observe(lambda x: self.update_color_range(),
                                      names="value")

        cd['color_map'] = widgets.Dropdown(
            options=sorted(COLORMAPS.keys()),
            value='3w_gby',
            description='Color Map',
            disabled=False,
        )
        cd['color_map'].observe(lambda x: self.update_cmap(), names="value")

        self.controls = widgets.HBox([
            widgets.VBox([
                cd['t_range'], cd['velocity_range'], cd['intensity_range']
            ]),
            widgets.VBox([
                cd['color_map'],
            ])
        ])

    def range(self, var):
        if var in self.individual_controls:
            return self.individual_controls[var].range
        return None

    def do_update(self, what):
        self.update_spectrogram()

    def update_spectrogram(self):
        """
        Recompute and display everything
        """
        sg = self.spectrogram
        tmin, tmax = self.range('t_range')
        # if the time range hasn't changed, we don't need to recompute?
        if (tmin, tmax) != (sg.t_start, sg.t_end):
            self.spectrogram = Spectrogram(self.digfile, tmin, tmax)
            sg = self.spectrogram


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
            sg.time * 1e6, sg.velocity, sg.intensity)

        self.colorbar = self.fig.colorbar(self.image, ax=self.axSpectrogram)

        self.axSpectrogram.set_title(self.title)
        self.axSpectrogram.set_xlabel('Time ($\mu$s)')
        self.axSpectrogram.set_xlim(tmin * 1e6, tmax * 1e6)
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
        self.spectrum(t)

    def spectrum(self, the_time):
        if the_time == None:
            # Initialize the axes
            self.axSpectrum.plot([0, 1], [0, 1], 'r-')
        else:
            the_spectrum = Spectrum(
                self.digfile.values(the_time, 8192),
                self.digfile.dt,
                remove_dc=True)
            line = self.axSpectrum.lines[0]
            intensities = the_spectrum.db
            line.set(xdata=intensities, ydata=the_spectrum.velocities)
            self.axSpectrum.set_xlim(
                (np.min(intensities), np.max(intensities)))


if __name__ == '__main__':
    pass  # unittest.main()
