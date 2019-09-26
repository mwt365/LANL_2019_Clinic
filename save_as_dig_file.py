#!/usr/bin/env python3
# coding: utf-8

import os
import numpy as np


def save_as_dig(filename, vvals, datatype, dt=20e-12, initialTime=0, voltageMultiplier= 6.103516e-5, voltageOffset=0):
    """
        Inputs:
            filename: string/path object pointing to where the file is.
            vvals: float/int array of voltage values for the datapoints.
            datatype: format of the data to input (save as unsigned bytes, signed ints or signed longs)
                The legal datatypes are unsigned bytes, signed 16 bits, and signed 32 bits.
            dt: float for the sampling speed of the oscilloscope you are simulating.
            initialTime: float describing the time of the first sample.
            voltageMultiplier: float for the conversion factor to when reading the data back.
            voltageOffset: float for the voltage offset on the detector.


        Output:
            1024 ascii header with last 512 bytes containing the notes about how the data
            is stored.

            The rest of the file will be writing the datapoints in binary.
    """
    dataformat = ""
    if datatype == '8' or datatype == np.dtype("ubyte"):
        dataformat = '8'
        datatype = np.dtype("ubyte")
    elif datatype == '16' or datatype == np.dtype("int16"):
        dataformat = '16'
        datatype = np.dtype("int16")
    elif datatype == '32' or datatype == np.dtype("int32"):
        dataformat = '32'
        datatype = np.dtype("int32")
    else:
        raise ValueError("datatype corresponds to an unsupported data type.")

    nsamples = len(vvals)

    # Now write the header, the important parts of which are
    # nsamples, bits, dt, t0, dv, v0
    with open(filename, 'w') as f: # This will fail if the file is not already created.
        f.write(" " * 512)  # write 512 bytes of spaces
        stuff = "\r\n".join([
            "Fri Sep 20 08:00:00 2019", # todays date Day Mon NumDay HH:MM:SS: YEAR
            nsamples, # the number of samples used.
            dataformat, # format for the data to be read as.
            dt, # Time step between datapoints.
            initialTime,
            voltageMultiplier,
            voltageOffset # Voltage zero
        ])
        f.write(stuff + "\r\n")
        f.write(" " * (510 - len(stuff)))
        f.close()
    with open(filename, 'ab') as f:
        for val in vvals:           
            vals = np.asarray(amp * seg, dtype=datatype)
            f.write(vals.tobytes())
        f.close()

if __name__ == '__main__':
    make_dig_file(
        "test.dig",
        [0, 1e-5, 2e-5, 4e-5],
        [1000, 1000, 5000, 3000]
    )