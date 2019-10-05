#!/usr/bin/env python3
# coding: utf-8

import os
import numpy as np


def save_as_dig(filename, vvals, datatype, dt=20e-12, initialTime=0,
                voltageMultiplier=6.103516e-5, voltageOffset=0):
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
    if isinstance(datatype, type(str("example"))
                  ) or isinstance(datatype, type('8')):
        if datatype == '8':
            dataformat = '8'
            datatype = np.dtype("ubyte")
        elif datatype == '16':
            dataformat = '16'
            datatype = np.dtype("int16")
        elif datatype == '32':
            dataformat = '32'
            datatype = np.dtype("int32")
        else:
            raise ValueError(
                "datatype corresponds to an unsupported data type.")
    elif datatype == np.dtype("ubyte"):
        dataformat = '8'
    elif datatype == np.dtype("int16"):
        dataformat = '16'
    elif datatype == np.dtype("int32"):
        dataformat = '32'
    else:
        raise ValueError("datatype corresponds to an unsupported data type.")

    nsamples = len(vvals)

    # Now write the header, the important parts of which are
    # nsamples, bits, dt, t0, dv, v0
    # This will fail if the file is not already created.
    with open(filename, 'w') as f:
        f.write(" " * 512)  # write 512 bytes of spaces
        stuff = "\r\n".join([
            "Fri Sep 20 08:00:00 2019",  # todays date Day Mon NumDay HH:MM:SS: YEAR
            str(nsamples),  # the number of samples used.
            str(dataformat),  # format for the data to be read as.
            str(dt),  # Time step between datapoints.
            str(initialTime),
            str(voltageMultiplier),
            str(voltageOffset)  # Voltage zero
        ])
        f.write(stuff + "\r\n")
        f.write(" " * (510 - len(stuff)))
        f.close()
    with open(filename, 'ab') as f:
        for val in vvals:
            vals = np.array(val, dtype=datatype)
            f.write(vals.tobytes())
        f.close()
