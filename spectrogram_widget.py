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

from spectrogram import Spectrogram


class PercentSlider(widgets.IntRangeSlider):
    """
    A Slider to represent the percentage of a range that we
    wish to display. 
    """

    def __init__(self, description, initial_value, yrange):
        super().__init__(
            description=description,
            min=0, max=100, readout=True,
            initial_value=initial_value,
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

    def __init__(self, spectrogram, **kwargs):
        """
        blah
        """
        assert isinstance(
            spectrogram, Spectrogram), "You must pass in a Spectrogram"
        sg = self.spectrogram = spectrogram
        self.title = sg.filename.split('/')[-1].replace("_", "\\_")
        # handle the keyword arguments here

        # Compute the base spectrogram (do we really need this?)
        self.sgram = sg.spectrogram(sg.t0, sg.t_final)

        # Create the figure to display this spectrogram
        gspec = {
            'width_ratios': [1.5, 6],
            'height_ratios': [5, ],
        }
        self.fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True,
                                      squeeze=True, gridspec_kw=gspec)
        self.axSpectrum, self.axSpectrogram = axes

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
        sg = self.spectrogram

        cd['t_range'] = PercentSlider(
            "Time Range", (0, 25), (sg.t0, sg.t_final))
        cd['t_range'].observe(
            lambda x: self.update_spectrogram(), names="value")

        cd['velocity_range'] = PercentSlider(
            "Velocity Range",
            (0, 50),
            (0.0, sg.v_max)
        )
        cd['velocity_range'].observe(lambda x: self.update_velocity_range(),
                                     names="value")

        cd['intensity_range'] = PercentSlider(
            "Color Range",
            (40, 70),
            (0, 1)  # really needs to get updated!
        )
        cd['intensity_range'].observe(lambda x: self.update_color_range(),
                                      names="value")
        self.controls = widgets.HBox([
            widgets.VBox([
                cd['t_range'], cd['velocity_range'], cd['intensity_range']
            ])
        ])

    def range(self, var):
        if var in self.individual_controls:
            return self.individual_controls[var].range
        return None

    def update_spectrogram(self):
        """
        Recompute and display everything
        """
        sg = self.spectrogram
        tmin, tmax = self.range('t_range')
        self.sgram = sg.spectrogram(tmin, tmax)  # this needs many more flags
        # Having recomputed the spectrum, we need to set the yrange
        # of the color map slider
        cmin, cmax = np.min(self.sgram['spectrogram']), np.max(
            self.sgram['spectrogram'])
        self.individual_controls['intensity_range'].range = (cmin, cmax)

        # if we have already displayed an image, remove it
        if self.image:
            self.axSpectrogram.remove(self.image)
        if self.colorbar:
            self.axSpectrogram.remove(self.colorbar)
        self.image = self.axSpectrogram.pcolormesh(
            self.sgram['t'] * 1e6,  # time, in microseconds
            self.sgram['v'],
            self.sgram['spectrogram']  # need to deal with colormap
        )
        self.colorbar = self.fig.colorbar(self.image, ax=self.axSpectrogram)

        self.axSpectrogram.set_title(self.title)
        self.axSpectrogram.set_xlabel('Time ($\mu$s)')
        self.axSpectrum.set_ylabel('Velocity (m/s)')

    def update_velocity_range(self):
        vmin, vmax = self.range('velocity_range')
        self.axSpectrogram.set_ylim(vmin, vmax)
        self.axSpectrum.set_ylim(vmin, vmax)

    def update_color_range(self):
        self.image.set_clim(self.range('intensity_range'))

    def plotter(self):
        """
        Generate an interactive spectrogram object inside a jupyter notebook.
        We will put the control panel on top, and the spectrogram underneath.
        Controls:
            widgets.IntSlider(min=0, max=100, value=20, readout=True)
            widgets.Text(value='blah', disabled=True, )
            widgets.BoundedFloatText(...)
            widgets.ToggleButton
            widgets.Checkbox
            widgets.HBox(list of widgets)
        """

        # we will store the widgets in a dictionary inside the spectrogram object
        self.widgets = {}
        w = self.widgets

        self.fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=(10, 10))
        self.axes = (ax0, ax1)
        sgram = self.spectrogram(self.t0, self.t_final,
                                 floor=0, normalize=True, remove_dc=True)
        self.im = ax0.pcolormesh(sgram['t'] * 1e6,  # convert to microseconds
                                 sgram['v'],
                                 sgram['spectrogram']
                                 )
        self._sgram = sgram
        self.cbar = self.fig.colorbar(self.im, ax=ax0)
        title = self.filename.split('/')[-1]
        ax0.set_title(title.replace("_", "\\_"))
        ax0.set_xlabel('Time ($\mu$s)')
        ax0.set_ylabel('Velocity (m/s)')

        # Let's handle the time range with percentages of the full range
        w['tRange'] = widgets.IntRangeSlider(
            description="Time Range",
            min=0, max=100,
            value=(0, 50),
            readout=True,
            width=400
        )
        # Handle the range of the colormap in the same way?
        w['velRange'] = widgets.IntRangeSlider(
            description="Velocity Range",
            min=0, max=100,
            value=(0, 50),
            readout=True,
            width=300
        )
        w['zRange'] = widgets.IntRangeSlider(
            description="Color Range",
            min=0, max=100,
            value=(40, 80),
            readout=True,
            width=400
        )
        w['controls'] = widgets.HBox(
            [widgets.VBox([
                w['tRange'],
                w['velRange'],
                w['zRange']
            ]),
            ])

        # Now set up the linkages

        def remake_image(self):
            """Use values from all the controls to remake the image"""

            ax0.remove(self.im)
            ax0.remove(self.cbar)
            self.im = ax0.pcolormesh(sg['t'] * 1e6,  # convert to microseconds
                                     sg['v'],
                                     sg['spectrogram']
                                     )

        def handle_vel_range(change):
            vels = self._sgram['v']
            vals = percentages(change.new, (vels[0], vels[-1]))
            ax0.set_ylim(*vals)

        def handle_z_range(change):
            vals = self._sgram['spectrogram']
            full_limits = np.min(vals), np.max(vals)
            z_limits = percentages(change.new, full_limits)
            self.im.set_clim(z_limits)

        def handle_t_range(change):
            t_limits = percentages(change.new, (self.t0, self.t_final))
            self._sgram = self.spectrogram(t_limits[0], t_limits[1])
            self.remake_image()

        w['velRange'].observe(handle_vel_range, names='value')
        w['zRange'].observe(handle_z_range, names='value')
        w['tRange'].observe(handle_t_range, names='value')
        display(w['controls'])


if __name__ == '__main__':
    pass  # unittest.main()
