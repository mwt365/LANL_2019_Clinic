#!/usr/bin/env python3
# coding:utf-8
"""
  Author:  LANL Clinic 2019 --<lanl19@cs.hmc.edu>
  Purpose: Analyze a spectrogram using kmeans to create a suitable
           multisegment colormap.
  Created: 10/17/19
"""

import numpy as np
from scipy.cluster.vq import kmeans, kmeans2
from spectrogram import Spectrogram
from plotter import COLORMAPS


def make_spectrogram_color_map(spectrogram, number_of_bands, name):
    """
    Given an instance of Spectrogram, attempt to cluster the
    values into n bands from which a colormap can be generated.
    If number_of_bands is 0, look for the most successful
    number of bands for kmeans up to 7.
    """
    assert isinstance(spectrogram, Spectrogram)
    assert isinstance(number_of_bands, int) and number_of_bands > 1
    assert isinstance(name, str)

    def calculate_centroids():
        vals = spectrogram.intensity.flatten()
        form = spectrogram.form
        if form == 'db':
            vals = np.power(10.0, 0.05 * vals)
        elif form == 'log':
            vals = np.power(10.0, vals)
        average_power = vals.mean()
        vals = np.log10(average_power + vals)

        vmin, vmax = vals.min(), vals.max()

        # now run the kmeans2 routine for a decreasing number of clusters
        kmns = kmeans2(vals, np.linspace(vmin, vmax, number_of_bands))
        centroids = np.array([vmin] + list(kmns[0]) + [vmax])
        if form == 'db':
            return centroids * 20.0
        if form == 'power':
            return np.power(10.0, centroids)
        return centroids

    def normalize_bounds(centroids):
        "Map the list of centroids to the range 0 to 1"
        boundaries = np.array(centroids)
        boundaries -= centroids[0]
        boundaries /= boundaries[-1]
        return boundaries

    def build_cdict(bounds):
        """Create a color dictionary that can be passed to
        matplotlib.colors.LinearSegmentedColormap to generate
        the colormap."""
        primaries = ('red', 'green', 'blue')
        cdict = {x: [(0.0, 1.0, 1.0)] for x in primaries}
        # What's a nice progression of colors?
        # white to yellow at the bottom end, fading to
        # red/brown, then to green, blue, purple, black
        loc = iter(bounds[2:-1])

        def add_row(x, r, g, b):
            for val, name in zip((r, g, b), primaries):
                cdict[name].append((x, val, val))
                if x < 0.98:
                    cdict[name].append((x + 0.005, 1, 1))

        try:
            add_row(next(loc), 0.9, 0.8, 0.5)  # to faded yellow
            add_row(next(loc), 0.5, 0, 0.5)
            add_row(next(loc), 0, 0.5, 0)
            add_row(next(loc), 1, 0.75, 0.2)
            add_row(next(loc), 0.2, 0.2, 0.2)
            add_row(next(loc), 0, 0, 0)
        except:
            pass

        add_row(1, 0, 0, 0)
        return cdict

    roids = calculate_centroids()
    bounds = normalize_bounds(roids)
    cdict = build_cdict(bounds)

    from matplotlib.colors import LinearSegmentedColormap
    COLORMAPS[name] = LinearSegmentedColormap(name, cdict)
    return dict(name=name, centroids=roids, cmap=COLORMAPS[name])


if __name__ == '__main__':
    from digfile import DigFile
    df = DigFile('../dig/sample.dig')
    sg = Spectrogram(df)
    roids = make_spectrogram_color_map(sg, 4, "Kitty")


