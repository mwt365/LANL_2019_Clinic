# coding:utf-8
"""
::

  Author:  LANL 2019 clinic --<lanl19@cs.hmc.edu>
  Purpose: To use k-means to look for gross features of a spectrogram
  Created: 11/25/19
"""

import numpy as np
import matplotlib.pyplot as plt

from ProcessingAlgorithms.preprocess.digfile import DigFile
from spectrogram import Spectrogram
from ProcessingAlgorithms.spectrum import Spectrum
from plotter import COLORMAPS
from scipy.cluster.vq import kmeans, kmeans2, whiten


def horizontal_lines(in_array: np.ndarray, threshold_fraction: float = 0.33):
    """
    Can we detect horizontal lines in a spectrogram?
    One strategy: count what fraction of columns have a local
    maximum in the vertical direction. This should work for
    horizontal lines of width one pixel.

    Input:
        in_array: 2-d numpy array
        threshold_fraction: fraction of pixels along a row that must
            be larger than the pixel above and below for the row to
            be considered a horizontal line.

    Outputs:
        an array of row indices corresponding to lines
        a copy of the 2-d input array with the lines removed by averaging
            the values above and below the rows determined to have lines
    """
    up = np.sign(in_array - np.roll(in_array, 1, axis=0))
    down = np.sign(in_array - np.roll(in_array, -1, axis=0))
    both = up + down
    # Now we'd like to tally the number of "2"s across the row
    both[both < 2] = 0
    collapse = np.mean(both, axis=1)

    # Collapse now holds the twice the probability that a pixel
    # on the row is a local maximum in the vertical direction
    # greater than 1 might be interesting
    collapse[collapse < 0.5 * threshold_fraction] = 0
    interesting = np.nonzero(collapse)[0]
    # Now remove the lines by averaging
    fixed = in_array.copy()
    for interest in interesting:
        try:
            fixed[interest, :] = (in_array[interest - 1, :] +
                                  in_array[interest + 1, :]) * 0.5
        except:
            pass
    return interesting, fixed


class GrossFeatures:
    """

    """

    def __init__(self, spec: Spectrogram, time_chunk: int = 16, velocity_chunk: int = 16):
        self.spectrogram = spec
        self.time_chunk = time_chunk
        self.velocity_chunk = velocity_chunk
        powers = spec.power(spec.intensity)
        nrows, ncols = powers.shape
        nrows = nrows // velocity_chunk
        ncols = ncols // time_chunk
        self.intensity = ar = np.zeros((nrows, ncols))
        rows = np.linspace(0, nrows * velocity_chunk,
                           num=nrows + 1, dtype=np.uint32)
        cols = np.linspace(0, ncols * time_chunk,
                           num=ncols + 1, dtype=np.uint32)
        for row in range(nrows):
            rfrom, rto = rows[row], rows[row + 1]
            for col in range(ncols):
                ar[row, col] = np.mean(
                    powers[rfrom:rto, cols[col]:cols[col + 1]])
        self.time = spec.time[0:len(spec.time):time_chunk]
        self.velocity = spec.velocity[0:len(spec.velocity):velocity_chunk]
        vals = ar.flatten()
        vals.sort()
        self.spower = vals
        self.threshold = vals[int(len(vals) * 0.8)]

    def show(self, fig=None):
        if fig == None:
            fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        im = ax.pcolormesh(self.time * 1e6, self.velocity,
                           20 * np.log10(self.intensity))
        im.set_cmap(COLORMAPS['3w_gby'])
        fig.colorbar(im, ax=ax)
        plt.show()

    def k_means(self, k: int = 5):
        """blah"""
        vals = self.intensity.flatten()
        vals = np.log10(vals) * 20.0
        guesses = np.linspace(0.0, vals.max(), k)
        centroid, label = kmeans2(vals, guesses)
        squashed = label.reshape(self.intensity.shape)
        interesting, fixed = horizontal_lines(squashed)

        fig = plt.figure()
        ax1 = fig.add_subplot(1, 3, 1)
        im1 = ax1.pcolormesh(self.time * 1e6, self.velocity,
                             20 * np.log10(self.intensity))
        im1.set_cmap(COLORMAPS['3w_gby'])
        fig.colorbar(im1, ax=ax1)

        int2, fixed2 = horizontal_lines(20 * np.log10(self.intensity))

        ax2 = fig.add_subplot(1, 3, 2)
        im2 = ax2.pcolormesh(self.time * 1e6, self.velocity,
                             fixed2)
        im2.set_cmap(COLORMAPS['3w_gby'])
        fig.colorbar(im2, ax=ax2)
        ax3 = fig.add_subplot(1, 3, 3)
        im3 = ax3.pcolormesh(self.time * 1e6, self.velocity, fixed)
        im3.set_cmap(COLORMAPS['3w_gby'])
        fig.colorbar(im3, ax=ax3)
        plt.show()


if __name__ == '__main__':
    sp = Spectrogram('../dig/PDV_CHAN1BAK001')
    gf = GrossFeatures(sp, time_chunk=8)
    gf.k_means()



