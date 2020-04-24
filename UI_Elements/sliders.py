# coding:utf-8
"""
::

  Author:  LANL 2019 clinic --<lanl19@cs.hmc.edu>
  Purpose: To represent sliders for the SpectrogramWidget
  Created: 03/6/20
"""

import ipywidgets as widgets


class ValueSlider(widgets.FloatRangeSlider):
    """
    A Slider to represent the range that we
    wish to display. The slider maps linearly over the range
    specified by yrange = (ymin, ymax). If a multiplier different
    from 1 is given, then values displayed by the widget
    will be divided by the multiplier to yield "real" values.
    So, for example, if "real" time values are in seconds, but
    we wish to display microseconds, the multiplier is 1e6 and
    values used internally get divided by 16 to yield true
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
