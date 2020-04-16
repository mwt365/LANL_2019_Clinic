"""
Something

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft


class Spectrum:
    """
    Given an array of values corresponding to equally spaced
    samples at sampling period *dt*, compute a power spectrum
    with optional windowing and conversion of the frequency
    axis to velocity.
    """

    def __init__(self, vals, dt, remove_dc=False, **kwargs):
        self.values = vals
        self.dt = dt
        self.n = len(vals)
        self.window = 'hann'
        self.remove_dc = remove_dc
        self.wavelength = 1550e-9  # very sketchy here
        for k, v in kwargs.items():
            setattr(self, k, v)
        self._compute()

    def _compute(self):
        raw = self.values
        if self.remove_dc:
            raw -= np.mean(raw)
        if self.window == 'hann':
            raw *= np.sin(np.linspace(0, np.pi, self.n))
        cspec = fft(raw)[0:1 + self.n // 2]
        self._power = np.power(np.abs(cspec), 2)
        self._frequencies = np.linspace(0.0, 0.5 / self.dt, len(cspec))

    @property
    def power(self):
        return self._power

    @property
    def frequencies(self):
        return self._frequencies

    @property
    def velocities(self):
        return self._frequencies * 0.5 * self.wavelength

    @property
    def db(self):
        if hasattr(self, 'epsilon'):
            return 10.0 * np.log10(self._power + self.epsilon)
        return 10.0 * np.log10(self._power)

    def plot(self, use_db=True, against_v=True):
        x = (self.velocities,
             "v (m/s)") if against_v else (self.frequencies, "Frequency (Hz)")
        y = (self.db, "Power (dB)") if use_db else (self.power, "Power")
        plt.plot(x[0], y[0])
        plt.xlabel(x[1])
        plt.ylabel(y[1])

        plt.show()

    def save(self):
        """
        We should provide a sensible way to export. What file formats
        should we support?
        """
        pass
