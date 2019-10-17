#! /usr/bin/env python3

from spectrogram import Spectrogram
import matplotlib.pyplot as plt


def seamSignalExtraction(intensityMatrix:list, startTime:int, stopTime:int, width:int, bottomIndex:int, metricFunction = None):
    """
        Input:
            intensityMatrix: a 2d array [velocity][time] and each cell 
                corresponds to the intensity at that point.
            startTime: index corresponding to the start time in the intensity 
                matrix of the signal being considered.
            stopTime: index corresponding to the stop time in the intensity
                matrix of the signal being considered.
            width:
                How wide of a window that is being considered at
                    each time step. The window will be split evenly up and down
            bottomIndex: Upper bound on the velocity. 
                Assumed to be a valid index for sgram's intensity array.
            metricFunction: function that will be minimized to determine the most 
                well connected signal. 
                Defaults to None:
                    minimizing the Manhattan distance between neighboring pixels.
                Needs to take two numbers and return a number.
    Output:
        list of indices that should be plotted as the signal as an array of indices
        indicating the intensity values that correspond to the most 
        connected signal as defined by the metric.
    """

    if stopTime <= startTime:
        raise ValueError("Stop Time is assumed to be greater than start time.")

    if metricFunction != None:
        # Check if metricFunction is a function that takes in two numbers and returns a number
        if not callable(metricFunction):
            raise ValueError("metricFunction must either be None or a function \
                that takes two numbers and returns a number.")
        raise TypeError("Method has not been implemented to take arbitrary functions.")
    else:
        numTimeSteps = stopTime - startTime
        velocities = np.zeros(numTimeSteps)
        # This will hold the answers that we are looking for to return.

        startIndex = np.argmax(intensityMatrix[:][startTime])

        top = startIndex + round((width/2)(numTimeSteps)+0.01) 
            # to get it to round up to 1 if it is 0.5
        bottomDP = max(startIndex - round((width/2)(numTimeSteps)+0.01), bottomIndex + 1) 
            # to get it to round up to 1 if it is 0.5


        DPTable = np.zeros((top - bottomDP))
        workingTable = np.zeros((top - bottomDP))
        parentTable = [[] for i in range(len(workingTable))] # This will be a list of lists.

        for timeIndex in range(stopTime-1, startTime-1, -1):
            for velocityIndex in range(0, top+1-bottomDP):
                bestSoFar = np.Infinity
                bestPointer = None
                for testIndex in range(max(velocityIndex - round(width/2 + 0.01), bottomDP), min(top, velocityIndex + round(width/2+0.01)+1)):
                    testIndex -= bottomDP
                    current = np.abs(intensityMatrix[testIndex][timeIndex+1] - intensityMatrix[velocityIndex][timeIndex]) + DPTable[testIndex]
                    if current < bestSoFar:
                        bestSoFar = current
                        bestPointer = testIndex
                workingTable[velocityIndex] = bestSoFar
                myPredecessors = parentTable[velocityIndex]
                parentTable[velocityIndex] = [bestPointer] + myPredecessors 
            DPTable = workingTable
            workingTable = np.zeros((top - bottomDP))

        # Now for the reconstruction.
        currentPointer = startIndex - bottomDP
        velocities = parentTable[currentPointer]
        # correct for bottom values.
        for i in range(len(velocities)):
            velocities[i] = velocities[i] + bottomDP

        print("Here is the last column of the full DP Table for debugging purposes")

        print(DPTable)

        return velocities



baseline_velocity_index = 412
filename = 'GEN3CH_4_009.dig'
sp_nl = Spectrogram(filename)

sgram_nl = sp_nl.spectrogram_no_log(0, 50e-6)
start = sp_nl.point_number(11.3)
stop = sp_nl.point_number(28)
print(stop - start)
maxVel = 1000
width = 5

signalSeamIndicies = seamSignalExtraction(sgram_nl['spectrogram'][:maxVel][:], start, stop, width, baseline_velocity_index)
signalSeamVelocities = sgram_nl['v'][signalSeamIndicies]

plt.scatter(sgram_nl['t'][start:stop+1], signalSeamVelocities)
