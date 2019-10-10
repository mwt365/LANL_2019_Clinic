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
