#!/usr/bin/env python3
# coding:utf-8
"""
  Author:  Peter N. Saeta --<saeta@hmc.edu>
  Purpose: To generate a sample .dig file with known parameters
  Created: 09/19/19
"""

import os
import numpy as np


def make_dig_file(filename, tvals, vvals, dt=20e-12):
    """
    Create a dig file with piecewise linear segments
    
    Inputs:
        filename: 
        tvals: array of times in ascending order
        vvals: array of corresponding velocities
        dt: time step, in seconds
    """
    nsamples = 1 + int((tvals[-1] - tvals[0]) / dt)
    dvdt = [(vvals[n + 1] - vvals[n]) / (tvals[n + 1] - tvals[n])
            for n in range(len(tvals) - 1)]
    # Let's compute the phase first
    # The frequency for given v is 2 v / λ
    # so the angular frequency is 4 π v / λ
    # For a segment that is ramping from v0 to v1 over an interval Δt,
    # the phase is a parabolic function of time.
    segments = []
    wavelength = 1550e-9  # m
    t, nsamp = tvals[0], 0
    phi0 = 0.0
    for nsegment in range(len(tvals) - 1):
        times = np.arange(0.0, tvals[nsegment + 1] - tvals[nsegment], dt)
        phis = phi0 + 4.0 * np.pi / wavelength * (
            vvals[nsegment] * times +
            0.5 * dvdt[nsegment] * times * times)
        segments.append(np.sin(phis))
    # Now write the header, the important parts of which are
    # nsamples, bits, dt, t0, dv, v0
    with open(filename, 'w') as f:
        f.write(" " * 512)  # write 512 bytes of spaces
        stuff = "\r\n".join([
            "Fri Sep 20 08:00:00 2019",
            f"{nsamples}",
            "16",
            f"{dt}",
            "0.0",
            "6.103516e-5",
            "0.0"
        ])
        f.write(stuff + "\r\n")
        f.write(" " * (510 - len(stuff)))
    with open(filename, 'ab') as f:
        for seg in segments:
            # pick a random amplitude, generate the sinusoids
            # corresponding to the phases in the segment, and
            # write to the file
            amp = np.random.randint(10, 2 ** 15)
            vals = np.asarray(amp * seg, dtype=np.int16)
            f.write(vals.tobytes())


if __name__ == '__main__':
    make_dig_file(
        "test.dig",
        [0, 1e-5, 2e-5, 4e-5],
        [8000, 8000, 5000, 3000]
    )
