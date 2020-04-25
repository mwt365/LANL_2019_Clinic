#!/usr/bin/env python3
# coding:utf-8
"""
::

  Author:  LANL Clinic 2019 --<lanl19@cs.hmc.edu>
  Purpose: To segment the data in a spectrogram based upon lack of signal.
  Created: 02/12/20
"""

import matplotlib.pyplot as plt
import numpy as np
import os
currDir = os.getcwd()
os.chdir(os.path.split(os.path.split((__file__))[0])[0])

from spectrogram import Spectrogram
from ProcessingAlgorithms.SaveFiles.save_as_dig_file import save_as_dig
os.chdir(currDir)

def intensityOverTime(Intensity, time, title:str):
    """
        Input: Intensity - 2d array of intensities [v, t]
                time - array of times
                title - string corresponding to the title of the graph.
        Effect:
            Plot the totalIntensity at each time slice.
    """
    
    totalInten = np.sum(Intensity, axis=0)
    
    plt.plot(time*1e6, totalInten)
    plt.title(title.replace("_", "\\_"))
    plt.xlabel(r'Time ($\mu$s)')
    plt.ylabel("Total Intensity across all velocities")
    plt.show()


def threshold(totalInten, fracOfMax):
    """
        Return the time indices that contain total intensity less than 
        fracOfMax of the maximum throughout the whole spectrogram.
    """

    maxValue = np.max(totalInten)

    inds = np.where(totalInten <= fracOfMax*maxValue)

    return sorted(inds[0])


def split(indicies:np.array, timeArray, timeBetweenSlices):
    """
        Input:
            indicies - sorted array of indicies that have been highlighted as 
                part of the rail which occurs during probe destruction.
            timeArray - the array of times used by the spectrogram.
            timeBetweenSlices - how long should I wait to split from one of the starts in $\mu$s. 
                This value will get converted to seconds.
        Output:
            tuple of Array times corresponding to the times for each segment.    
    """
    time = timeArray.copy()
    timeValues = time[indicies].copy()*1e6
    timeBetweenSlices *= 1e6 # Convert to microseconds.
    data = []
    ind = 0
    while ind < len(indicies):
        b = np.where(timeValues - timeValues[ind] >= timeBetweenSlices)[0] # To make it an array.
    
        data += [ind]
        if len(b) == 0:
            ind = len(indicies)
        else:
            ind = b[0] # Since the indicies will be sorted from np.where

    if len(data) == 0:
        raise RuntimeError("We should have set something back there")
   
    return np.array([indicies[data[i]] for i in range(len(data))])


def splitIntoDigFiles(SpectrogramObject:Spectrogram, fracOfMax = 0.1, timeBetweenSlices = 5e-5):
    intensity = SpectrogramObject.intensity
    totalInten = np.sum(intensity, axis = 0)
    totalInten -= np.min(totalInten)
    
    inds = threshold(totalInten, fracOfMax)

    timeForSplits = split(inds, SpectrogramObject.time, timeBetweenSlices)

    df = SpectrogramObject.data
    filelocation = df.filename
    # Strip the extension and then make the folder and start writing segments.
    parent, filename = os.path.split(filelocation)
    folder, _ = os.path.splitext(filename)
    folder = os.path.join(parent, folder)
    if not os.path.exists(folder):
        os.makedirs(folder)
    text = open(df.path, 'rb').read(512).decode('ascii')
    text = text.replace("\000", "").strip()
    header = "Segment {0:02d}\r\n" + text
    basename = "seg"
    # We are splitting based upon probe destruction which occurs at the end of the segments.
    segmentOffset = df.t0
    for n in range(len(timeForSplits)+1):
        head = header.format(n)
        try:
            t_stop = df.time_values(SpectrogramObject.time[timeForSplits[n]], 1)[0]
            print("spec time", SpectrogramObject.time[timeForSplits[n]], "and its corresponding point number", df.point_number(SpectrogramObject.time[timeForSplits[n]]))
        except:
            t_stop = df.t_final
        print("segmentOffset", segmentOffset, "tEnd", t_stop)
        print("The final time point is", df.t_final)
        vals = df.raw_values(segmentOffset, t_stop)
        name = f"{basename}{n:02d}.dig"
        kwargs = dict(
            dt=df.dt,
            initialTime=segmentOffset,
            voltageMultiplier=df.dV,
            voltageOffset=df.V0
        )
        save_as_dig(os.path.join(folder, name),
                        vals, df.data_format,
                        top_header=head, **kwargs)
        # df.extract(os.path.join(folder, name), t_start, t_stop)
        segmentOffset = t_stop
    return f"{len(timeForSplits)} files written in {folder}"

if __name__ == "__main__":
    currDir = os.getcwd()
    os.chdir(os.path.split(os.path.split((__file__))[0])[0])
    
    from ProcessingAlgorithms.preprocess.digfile import DigFile

    os.chdir(currDir)
    

    digFolder = DigFile.dig_dir()
    allDigs = DigFile.inventory()["file"] # Just the files that are not segments.

    saveLoc = os.path.join(os.path.split(digFolder)[0], "TotalIntensityMaps\\Median\\")
    if not os.path.exists(saveLoc):
        os.makedirs(saveLoc)
    
    fractions = np.arange(10)
    fractionsL = len(fractions)

    data = np.zeros((fractionsL, len(allDigs)))
    data[:,0] = fractions
    for i in range(len(allDigs)):
        filename = allDigs[i]
        path = os.path.join(digFolder, filename)
        spec = Spectrogram(path)
        inTen = np.sum(spec.intensity, axis = 0)
            # Offset the values so that everything is non negative.
        inTen = inTen - np.min(inTen)

        for j in range(fractionsL):
            frac = np.power(10.0, -1*fractions[j])
            data[j,i] = len(threshold(inTen, frac))
        
        del spec

    name = os.path.join(saveLoc, "NumberOfLowValuesVsFracOfMax.txt")

    np.savetxt(name, data)
