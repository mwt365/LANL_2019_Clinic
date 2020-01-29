#!/usr/bin/env python3
# coding:utf-8
"""
::

  Author:  LANL Clinic 2019 --<lanl19@cs.hmc.edu>
  Purpose: To identify baselines in a spectrogram
  Created: 10/10/19
"""

import numpy as np
import pandas as pd
from spectrogram import Spectrogram
from scipy.signal import find_peaks
from scipy.fftpack import fft
import random
import peak_follower


def baselines_by_squash(spectrogram: Spectrogram):
    """
    Return a list of baseline velocities and their uncertainties.

    Inputs:
      -  spectrogram: an instance of Spectrogram

    Outputs:
      -  peaks: an array of peaks, in descending order of strength
      -  widths: an array of uncertainty values for the peaks
      -  heights: corresponding peak heights, normalized to
            the greatest height

    Observations:
        This routine probably does a poor job of discriminating against
        lone peaks arising from an anomalously large value at a fairly
        isolated time. It might be better to chop the range into several
        intervals and insist on a significant peak in each.
    """
    assert isinstance(spectrogram, Spectrogram)
    # Collapse along the time axis, making sure to use power,
    # not dB
    powers = spectrogram.power(spectrogram.intensity)
    # We'd like to squash into about 8 distinct segments
    # along axis 1 and then make sure that we get consistent
    # values before calling a peak a baseline
    segments = 8
    boundaries = list(range(0, len(spectrogram.time),
                            len(spectrogram.time) // segments))
    boundaries.append(len(spectrogram.time))
    squash = []
    for n in range(segments):
        squash.append(
            powers[:, boundaries[n]:boundaries[n + 1]].mean(axis=1))

    combined_spectrum = spectrogram.power(
        spectrogram.intensity).sum(axis=1)
    tallest = combined_spectrum.max()
    peaks, properties = find_peaks(
        combined_spectrum,
        height=0.01 * tallest,  # peaks must be this tall to count
        distance=100,  # peaks must be separated by this much at minimum
    )
    heights = properties['peak_heights']
    peak_ht = heights.max()
    ordering = np.flip(np.argsort(heights))
    peak_positions = peaks[ordering]
    peaks = spectrogram.velocity[peak_positions]
    heights = heights[ordering] / peak_ht
    dv = spectrogram.velocity[1] - spectrogram.velocity[0]
    return peaks, np.ones(len(peaks)) * dv, heights


def baselines_by_fft(sgram):
    """
    Return a list of baseline velocities and their uncertainties.

    Inputs:
      -  spectrogram: an instance of Spectrogram

    Outputs:
      -  peaks: an array of peaks, in descending order of strength
      -  widths: an array of uncertainty values for the peaks
      -  heights: corresponding peak heights, normalized to
            the greatest height

    Observations:
        This routine probably does a poor job of discriminating against
        lone peaks arising from an anomalously large value at a fairly
        isolated time. It might be better to chop the range into several
        intervals and insist on a significant peak in each.
    """
    # print(type(sgram))
    # assert isinstance(sgram, Spectrogram)

    # Perform a single Fourier transform of the voltage data in the range
    # corresponding to this spectrogram, rounded down to the nearest power
    # of 2 (we could also zero pad up to the next higher power of two).
    first_point, last_point = sgram.data._points(None, None)
    nPoints = last_point - first_point + 1
    octaves = np.log2(nPoints)  # how many octaves are spanned
    num_points = int(2 ** np.floor(octaves))
    vals = sgram.data.values(None, num_points)
    # Now compute the power spectrum of these points
    # At present, we aren't worrying about a window function, assuming that
    # the noise spread across all frequencies from abrupt shifts will not
    # worry the peaks significantly.
    cspec = fft(vals)[0:1 + num_points // 2]
    powers = np.power(np.abs(cspec), 2)

    velocities = np.linspace(0.0, 0.25 / sgram.data.dt,
                             len(cspec)) * sgram.wavelength
    # At this point, powers vs velocities should show some very
    # narrow spikes. Let's normalize the array of powers
    peak_power = powers.max()
    powers /= peak_power

    peaks, properties = find_peaks(
        powers,
        height=0.001,   # peaks must be this tall to count
        # peaks must be separated by this much at minimum
        distance=len(powers) // 1024,
    )
    heights = properties['peak_heights']
    ordering = np.flip(np.argsort(heights))

    # sort the peaks and heights into descending order
    peaks, heights = peaks[ordering], heights[ordering]

    neighborhoods = []
    width = 50
    for peak_center in peaks:
        low, high = max(0, peak_center -
                        width), min(len(powers) - 1, peak_center + width)
        neighborhoods.append([velocities[low:high], powers[low:high]])
    return neighborhoods




if __name__ == '__main__':
    import os
    from digfile import DigFile

    path = "/Users/trevorwalker/Desktop/Clinic/For_Candace/newdigs"

    os.chdir(path)

    df = DigFile('CH_2_009.dig')

    baselines_v = []

    sgram = Spectrogram(df, 0.0, 60.0e-6, form='db')
    # hoods = baselines_by_fft(sgram)


    peaks, dvs, heights = baselines_by_squash(sgram) 



    df = pd.DataFrame(dict(peaks=peaks, heights=heights))

    print(df)


    # print(sgram.v_max)

    # for n, h in enumerate(hoods):
    #     max_v = 0
    #     max_i = 0
    #     print(f"Peak {n}\nVelocity{n}\tIntensity{n}")
    #     v, i = h
    #     for j in range(len(v)):
    #         print(f"{v[j]:.4f}\t{i[j]:.4f}")
    #         if i[j] > max_i:
    #             max_i = i[j]
    #             max_v = v[j]
                
    #     print("\n")
    #     # print("velocity:", max_v,"\nintensity:", max_i)
    #     baselines_v.append(max_v)


    # actual_baselines = []

    # for baseline in baselines_v:
    #     print("is there a baseline at: ", baseline, "?", end=" ")
    #     ans = input("(y/n)\n")
    #     if ans == "y":
    #         actual_baselines.append(baseline)
    #     else:
    #         continue

    # for baseline in actual_baselines:
    #     ans = input("where does the start begin? (microseconds)")
    #     try: 
    #         ans = int(ans)
    #     except:
    #         print("input can not be converted to integer")
    #         break

    #     if (ans > 0) and (ans < sgram.v_max):
    #         baseline_index = sgram._velocity_to_index(baseline)

    #         intensity_of_baseline = sgram.intensity[baseline_index][ans]
    #         threshold = .90
    #         threshold_intensity = threshold * intensity_of_baseline

    #         potential_starting_velos = find_start(sgram, ans, baseline_index, threshold_intensity)



    #         print( potential_starting_velos ) 
    #         print( len(sgram.signal_to_noise()) )

    # print(baselines_v)

