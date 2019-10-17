import numpy as np

def extractSignal(intensityMatrix, velocities, lowerVelocityThreshold, upperVelocityThreshold, plotAbove):
    """
    Input:
        sgram: a spectrogram object
        intensityMatrix: a 2d array [velocity][time] and each cell corresponds to the intensity at that point.
        velocities: a 1d array of the velocities being consider.
        lowerVelocityThreshold: what you consider to be the lower threshold on an 
        interesting velocity. 
            Assumed to be a valid index for sgram's intensity array.
        upperVelocityThreshold: Upper bound on the velocity. 
            Assumed to be a valid index for sgram's intensity array.
        plotAbove: keep everything above this fraction of the maximum.

    Output:
        list of indices that should be plotted as the signal as an array of indices
        indicating the highest intensity value at the given time slice.
    """
    
    time_then_velocity_intensities = np.transpose(intensityMatrix)

    signalVelocityIndecies = np.zeros(time_then_velocity_intensities.shape[0], dtype = np.int64)

    if time_then_velocity_intensities.shape[-1] != len(velocities):
        raise ValueError("The intensities matrix and the velocity matrix mismatch in shape.")

    dataPoints = []
    for i in range(len(signalVelocityIndecies)):
        signalVelocityIndecies[i] = np.argmax(time_then_velocity_intensities[i][lowerVelocityThreshold:upperVelocityThreshold+1]) + lowerVelocityThreshold
        AboveThresData = []
        for j in range(lowerVelocityThreshold, upperVelocityThreshold+1):
            if time_then_velocity_intensities[i][j] >= plotAbove * time_then_velocity_intensities[i][signalVelocityIndecies[i]]:
                AboveThresData.append(j)
        dataPoints.append(np.array(AboveThresData))
    
    print(dataPoints)

    return dataPoints

    
def seamExtraction(intensityMatrix, startTime, stopTime, width, bottomIndex, metricFunction = None):
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
        velocities = np.zeros((stopTime - startTime), dtype = np.int64)
        # This will hold the answers that we are looking for to return.

        startIndex = np.argmax(intensityMatrix[:][startTime])

        top = startIndex + round((width/2)(stopTime - startIndex)+0.01) 
            # to get it to round up to 1 if it is 0.5
        bottomDP = max(startIndex - round((width/2)(stopTime - startIndex)+0.01), bottomIndex + 1) 
            # to get it to round up to 1 if it is 0.5


        DPTable = np.zeros(((top - bottomDP)*(stopTime - startTime), stopTime - startTime + 1))
        parentTable = np.zeros(DPTable.shape(), dtype = np.int64)

        for timeIndex in range(stopTime-1, startTime-1, -1):
            for velocityIndex in range(bottomDP, top+1):
                bestSoFar = np.Infinity
                bestPointer = None
                for testIndex in range(max(velocityIndex - round(width/2 + 0.01), bottomDP+1), min(top, velocityIndex + round(width/2+0.01)+1)):
                    current = np.abs(intensityMatrix[testIndex][timeIndex+1] - intensityMatrix[velocityIndex][timeIndex]) + DPTable[testIndex][timeIndex+1]
                    if current < bestSoFar:
                        bestSoFar = current
                        bestPointer = testIndex
                DPTable[velocityIndex][timeIndex] = bestSoFar
                parentTable[velocityIndex][timeIndex] = bestPointer


        # Now for the reconstruction.
        currentPointer = startIndex
        for timeIndex in range(startTime, stopTime+1):
            velocities[timeIndex] = currentPointer
            currentPointer = parentTable[currentPointer][timeIndex]

        return velocities