# coding:utf-8
"""
::

  Author:  LANL Clinic 2019 --<lanl19@cs.hmc.edu>
  Purpose: Split a dig file into evenly timed segments with the left over at the end.
  Created: 04/23/20
"""



import os
import numpy as np
from collections import OrderedDict


from ProcessingAlgorithms.preprocess.digfile import DigFile
from ProcessingAlgorithms.SaveFiles.save_as_dig_file import save_as_dig


def splitIntoEvenFrames(myDig: DigFile, timeBetweenFrames:float = 50e-6, basename=""):
    """
    Prepare a subdirectory filled with the segments that are every timeBetweenFrames.
    If no basename is supplied, use the name of the source file before the
    .dig extension.
    """

    startTimes = []

    nextTime = myDig.t0
    while nextTime <= myDig.t_final:
        startTimes.append(nextTime)
        nextTime = startTimes[-1] + timeBetweenFrames

    # Now we have all the start times. Let's split the file into multiple.

    parent, file = os.path.split(myDig.path)
    folder, _ = os.path.splitext(file)
    if not basename:
        basename = "seg"
    # Make the directory
    home, filename = os.path.split(myDig.path)
    folder = os.path.join(parent, folder)
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Prepare the top 512-byte header
    myDig = myDig
    text = open(myDig.path, 'rb').read(512).decode('ascii')
    text = text.replace("\000", "").strip()
    the_date = myDig.date

    kwargs = dict(
        dt=myDig.dt,
        initialTime=0,
        voltageMultiplier=myDig.dV,
        voltageOffset=myDig.V0
    )
    for n in range(len(startTimes)):
        t_start = startTimes[n]
        n_start = int((t_start - myDig.t0) / myDig.dt)
        try:
            t_stop = startTimes[n + 1]
        except:
            t_stop = myDig.t_final
        heading = OrderedDict(
            Segment=n,
            n_start=n_start,
            t_start=f"{t_start:.6e}",
            t_stop=f"{t_stop:.6e}",
            date=the_date,
            header="\r\n" + text,
        )

        vals = myDig.raw_values(t_start, t_stop)
        name = f"{basename}{n:02d}.dig"
        save_as_dig(os.path.join(folder, name),
                    vals, myDig.data_format,
                    top_header=heading, date=the_date, **kwargs)
    return f"{len(startTimes)} files written in {folder}"