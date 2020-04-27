#!/usr/bin/env python3
# coding:utf-8
"""
::

  Author:  LANL 2019 clinic --<lanl19@cs.hmc.edu>
  Purpose: To provide convenient controls for the sliders in
    spectrogram_widget.py
  Created: 02/7/20
"""


import ipywidgets as widgets

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

