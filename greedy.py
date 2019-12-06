import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from digfile import DigFile

from spectrogram import Spectrogram

def greedy(spectogram,startUS,endUS,vstart,vend):
#     del times,velocities,intensities
    tstart = timeIndex(startUS,spectogram)
    tend = timeIndex(endUS,spectogram)
    
    
    times,velocites,intensities = spec.slice((spec.time[tstart],spec.time[tend]),(vstart,vend))
    greedyIndex = np.zeros(tend-tstart,dtype=int)
    
    #greedyVels is storing the index of the velocities
    greedyIndex[0] = int(np.argmax(intensities[:,0]))
#     print(greedyVels)
    i = 1
    while i != tend-tstart:
#         print(i)
#         print(int(greedyIndex[i-1]))
        greedyIndex[i] = greedyCompare(intensities[:,i],intensities[:,i+1],greedyIndex[i-1])
        i += 1
#     print( greedyIndex)
    Vcoords = np.zeros(tend-tstart)
    times = spectogram.time[tstart:tend+1]
#     print(coords)
    i = 0
    while i < tend-tstart:
#         print(i)
        Vcoords[i] = velocities[greedyIndex[i]]
#         print(i,greedyIndex[i],velocities[greedyIndex[i]])
        i += 1
    return times[:-1], Vcoords, velocities
        
        
        
        
        
        
def greedyCompare(time0intens, time1intens, start ):
    """ 
    Inputs: two arrays of intensities, at two times. indexes = velocity indexes
                start: the index in time0 of the current velocity selection
    Output: index of the greedy match
    verified 11/26
    """
#     print('start = ', start)
    oldVelocity = time0intens[start]
    nextOptions = []
    for i in range(0,3): #makes -1,0,1
        try:
            nextOptions += [abs(oldVelocity-time1intens[start+i])]
        except:
            nextOptions += [10000]
    return start+nextOptions.index(min(nextOptions))

def timeIndex(microS,spectrogram):
    seconds = microS*1e-6
    start = spectrogram.time[0]
    timeStep = abs(start-spectrogram.time[1])
    steps = (seconds-start)/timeStep
    steps = int(steps)
    return  steps


if __name__ == '__main__':
    sp = Spectrogram('GEN3CH_4_009.dig', None,
                     None, overlap_shift_factor=1 / 4)
    print(sp)