import numpy as np
from spectrogram import Spectrogram

def extractSignal(sgram, lowerVelocityThreshold, upperVelocityThreshold, plotAbove):
    """
    Input:
        sgram: a spectrogram object
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
    
    signalVelocityIndecies = np.zeros(len(sgram.t), dtype=int)

    time_then_velocity_intensities = np.transpose(sgram.intensity)

    for i in range(len(signalVelocityIndecies)):
        signalVelocityIndecies[i] = np.argmax(time_then_velocity_intensities[i][lowerVelocityThreshold:upperVelocityThreshold+1]) + lowerVelocityThreshold
    
    return signalVelocityIndecies
