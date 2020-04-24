#!/usr/bin/env python3
# coding: utf-8

import numpy as np
from datetime import datetime

def save_as_dig(filename, vvals, datatype, dt=20e-12, initialTime=0,
                voltageMultiplier=6.103516e-5, voltageOffset=0,
                top_header="", **kwargs):
    """
        Inputs:
            filename: string/path object pointing to where the file is.
            vvals: float/int array of voltage values for the datapoints.
            datatype: format of the data to input (save as unsigned bytes,
            signed ints or signed longs) The legal datatypes are
            unsigned bytes, signed 16 bits, and signed 32 bits.

            dt: float for the sampling speed of the
            oscilloscope you are simulating.

            initialTime: float describing the time of the first sample.

            voltageMultiplier: float for the conversion factor
            to when reading the data back.

            voltageOffset: float for the voltage offset on the detector.


        Output:
            1024 ascii header with last 512 bytes containing the notes
            about how the data is stored.

            The rest of the file will be writing the datapoints in binary.
    """
    dataformat = ""
    date = kwargs.get("date", "")
    if isinstance(datatype, type(str("example"))
                  ) or isinstance(datatype, type('8')):
        if datatype == '8':
            dataformat = '8'
            datatype = np.dtype("u1")
        elif datatype == '16':
            dataformat = '16'
            datatype = np.dtype("<u2")
        elif datatype == '32':
            dataformat = '32'
            datatype = np.dtype("<u4")
        else:
            raise ValueError(
                "datatype corresponds to an unsupported data type.")
    elif datatype == np.dtype("u1"):
    elif datatype == np.dtype("ubyte"):
        dataformat = '8'
    elif datatype == np.dtype("<u2"):
        dataformat = '16'
    elif datatype == np.dtype("<u4"):
        dataformat = '32'
    else:
        raise ValueError("datatype corresponds to an unsupported data type.")

    nsamples = len(vvals)

    # If the top_header argument is a dictionary, prepare a string
    # representation.
    if isinstance(top_header, dict):
        keys = top_header.keys()
        head = "\r\n".join([f"{k} = {top_header[k]}" for k in keys])
        top_header = head

    # Now write the header, the important parts of which are
    # nsamples, bits, dt, t0, dv, v0
    # This will fail if the file is not already created.

    with open(filename, 'w') as f:
        if len(top_header) < 512:
            top_header += " " * (512 - len(top_header))
        f.write(top_header)  # write 512 bytes of spaces
        stuff = "\r\n".join([
            # todays date Day Mon NumDay HH:MM:SS: YEAR
            # This is achieved using ctime() in the datetime library. 
            # Reference https://docs.python.org/3/library/datetime.html
            date,
            # the number of samples used.
            str(nsamples),
            # format for the data to be read as.
            str(dataformat),
            # Time step between datapoints.
            str(dt),
            str(initialTime),
            str(voltageMultiplier),
            str(voltageOffset)  # Voltage zero
        ])
        f.write(stuff + "\r\n")
        f.write(" " * (510 - len(stuff)))
        f.close()
    with open(filename, 'ab') as f:
        f.write(np.array(vvals, dtype=datatype).tobytes())
