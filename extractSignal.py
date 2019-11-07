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

    
def seamExtraction(intensityMatrix, startTime:int, stopTime:int, width:int, signalJump:int=50, bottomIndex:int=0, topIndex:int=None, metricFunction = None):
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
            signalJump:
                How far up you think the signal will be above the bottomIndex
                    defaults to 50
            bottomIndex: Upper bound on the velocity. 
                Assumed to be a valid index for sgram's intensity array.
                Defaults to zero
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
        if width % 2 == 1:
            width += 1

        tableHeight = stopTime - startTime + 1

        velocities = np.zeros(tableHeight, dtype = np.int64)
        # This will hold the answers that we are looking for to return.

        halfFan = width//2

        transposedIntensities = np.transpose(intensityMatrix) # It is assumed to start as velocities then time
        # I want time then velocities to find the argmax.

        if topIndex == None:
            topIndex = transposedIntensities.shape[0] # I just want the maximum velocity index
        if topIndex <= bottomIndex + signalJump:
            topIndex = min((stopTime - startTime + 1)*width + bottomIndex, transposedIntensities.shape[0])


        startIndex = np.argmax(transposedIntensities[startTime][bottomIndex+signalJump:topIndex+1])+bottomIndex+signalJump


        top = min(startIndex + (stopTime - startTime)*halfFan, topIndex)
            # to get it to round up to 1 if it is 0.5
        bottomDP = max(startIndex - (stopTime - startTime)*halfFan, bottomIndex + 1) 
            # to get it to round up to 1 if it is 0.5

        tableWidth = top - bottomDP+1
        
        print("Top", top)
        print("bottomDP", bottomDP)

        print("The starting of index is ", startIndex)

        print()

        print("t2-t1", stopTime-startTime)
        print("(t2-t1)*halfan", (stopTime-startTime)*halfFan)

        print()

        print(topIndex)
        print(bottomIndex+1)

        print()

        print(tableWidth)
        print(tableHeight)

        DPTable = np.zeros((tableHeight, tableWidth))

        parentTable = np.zeros(DPTable.shape, dtype = np.int64)

        for timeIndex in range(2, tableHeight + 1):
            dpTime = tableHeight - timeIndex
            intensityTime = stopTime-timeIndex
            for velocityIndex in range(top, bottomDP-1, -1):
                bestSoFar = np.Infinity
                bestPointer = None
                for testIndex in range(-halfFan, halfFan+1): # The + 1 is so that you get a balanced window.
                    if velocityIndex + testIndex >= bottomDP and velocityIndex + testIndex <= top:
                        # then we have a valid index to test from.                
                        current = np.abs(intensityMatrix[testIndex+velocityIndex][intensityTime+1] - intensityMatrix[velocityIndex][intensityTime]) + DPTable[dpTime+1][velocityIndex+testIndex-bottomDP]
                        if current < bestSoFar:
                            bestSoFar = current
                            bestPointer = velocityIndex + testIndex - bottomDP                            
                DPTable[dpTime][velocityIndex-bottomDP] = bestSoFar
                parentTable[dpTime][velocityIndex-bottomDP] = bestPointer


                # print("Just finished computing the answer for (v,t): (", velocityIndex,",", dpTime, ")")
                # print("We are going backwards so there are", dpTime, "more columns to fill in the table")

        # Now for the reconstruction.
        currentPointer = np.argmin(DPTable[0])
        
        print("The value of the current pointer is", currentPointer)

        velocities = reconstruction(parentTable, currentPointer, bottomDP)

        return velocities, parentTable, DPTable


def reconstruction(parentTable, startPoint, bottomDP):
    tableHeight = int(parentTable.shape[0])
    velocities = np.zeros(tableHeight, dtype = np.int64)
    for timeIndex in range(tableHeight):
        velocities[timeIndex] = startPoint + bottomDP
        print(timeIndex)
        print("myStart", startPoint)
        startPoint = parentTable[timeIndex][startPoint]

    return velocities