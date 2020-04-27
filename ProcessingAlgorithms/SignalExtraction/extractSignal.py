import numpy as np
import matplotlib.pyplot as plt
# from scipy.spatial import distance
from spectrogram import Spectrogram
    
def seamExtraction(SpectrogramObject:Spectrogram, startTime:int, stopTime:int, width:int, signalJump:int=50, bottomIndex:int=0, topIndex:int=None, order:int = None):
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
            order: the order of the minkowski distance function that will be minimized 
                to determine the most well connected signal. 
                Defaults to 2:
                    minimizing the Euclidean distance between neighboring pixels.
                
    Output:
        list of indices that should be plotted as the signal as an array of indices
        indicating the intensity values that correspond to the most 
        connected signal as defined by the metric.
    """

    if stopTime <= startTime:
        raise ValueError("Stop Time is assumed to be greater than start time.")

    else:
        if width % 2 == 1:
            width += 1

        tableHeight = stopTime - startTime + 1

        velocities = np.zeros(tableHeight, dtype = np.int64)
        # This will hold the answers that we are looking for to return. It is a list of indices.

        halfFan = width//2

        if topIndex == None:
            topIndex = SpectrogramObject.velocity.shape[0] # I just want the maximum velocity index
        if topIndex <= bottomIndex + signalJump:
            topIndex = min((stopTime - startTime + 1)*width + bottomIndex, SpectrogramObject.velocity.shape[0])


        transposedIntensities = np.transpose(SpectrogramObject.intensity)
        startIndex = np.argmax(transposedIntensities[startTime][bottomIndex+signalJump:topIndex+1])+bottomIndex+signalJump

        bottomDP = max(startIndex - (stopTime - startTime)*halfFan, bottomIndex + 1) 

        print("BottomDP:", bottomDP, "TopIndex:", topIndex)

        t, vel, raw  = SpectrogramObject.slice((SpectrogramObject.time[startTime], SpectrogramObject.time[stopTime]), (SpectrogramObject.velocity[bottomDP], SpectrogramObject.velocity[topIndex]))

        print("new time shape", t.shape, "vel shape:", vel.shape)

        print("Before the transpose: raw.shape", raw.shape)
        raw = np.transpose(raw)
        tableHeight, tableWidth = raw.shape
        
        print("raw.shape = ",raw.shape)

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
            for velocityIndex in range(tableWidth):
                bestSoFar = np.Infinity
                bestPointer = None
                for testIndex in range(-halfFan, halfFan+1): # The + 1 is so that you get a balanced window.
                    if velocityIndex + testIndex >= 0 and velocityIndex + testIndex < tableWidth:
                        # then we have a valid index to test from.                
                        current = np.power(np.abs(raw[dpTime+1][testIndex+velocityIndex]-raw[dpTime][velocityIndex]), order) \
                         + DPTable[dpTime+1][velocityIndex+testIndex]
                        if current < bestSoFar:
                            bestSoFar = current
                            bestPointer = velocityIndex + testIndex                            
                DPTable[dpTime][velocityIndex] = bestSoFar
                parentTable[dpTime][velocityIndex] = bestPointer

        # Now for the reconstruction.
        currentPointer = np.argmin(DPTable[0])
        
        # Get the values of the DP table to a more reasonable values by taking the orderth root.
        # for timeIndex in range(tableHeight+1):
        #     for velocityIndex in range(tableWidth):
        #         DPTable[timeIndex][velocityIndex] = np.power(DPTable[timeIndex][velocityIndex], 1/order)

        print("The value of the current pointer is", currentPointer)

        print("The minimum cost is", DPTable[0][currentPointer])

        velocities = reconstruction(parentTable, currentPointer, bottomDP)

        return velocities, parentTable, DPTable, bottomDP, topIndex


def reconstruction(parentTable, startPoint, bottomDP):
    tableHeight = int(parentTable.shape[0])
    velocities = np.zeros(tableHeight, dtype = np.int64)
    for timeIndex in range(tableHeight):
        velocities[timeIndex] = startPoint + bottomDP
        print(timeIndex)
        print("myStart", startPoint)
        startPoint = parentTable[timeIndex][startPoint]

    return velocities

def normalize(timeVelocityIntensity):
    """
        Time Vs Velocity intensity values.

        Subtract out the min at each time step. Then, find
        the maximum at that time step. Normalize the intensity
        at velocity along this time step by the maximum intensity
        at this time step. Return the normalized array.
    """
    newArray = np.zeros(timeVelocityIntensity.shape, dtype = np.float32)
    for timeInd in range(timeVelocityIntensity.shape[0]):
        minValue = np.min(timeVelocityIntensity[timeInd])
        timeVelocityIntensity[timeInd] += -1*minValue
        maxValue = np.max(timeVelocityIntensity[timeInd])
        for velInd in range(timeVelocityIntensity.shape[1]):
            curr = timeVelocityIntensity[timeInd][velInd]
            newValue = curr/maxValue
            newArray[timeInd][velInd] = newValue

    return newArray

def mainTest(SpectrogramObject:Spectrogram, startTime:int, stopTime:int, signalJump:int=50, bottomIndex:int=0, topIndex:int=None, vStartInd: int=None):
    widths = [1,3,5,11,21]
    orders = [1,2,10]
    seam = []

    velocities = SpectrogramObject.velocity
    time = SpectrogramObject.time
    for width in widths:
        for order in orders:
            signal, p_table, dp_table, botVel, topVel = seamExtraction(SpectrogramObject, startTime, stopTime, width, signalJump, bottomIndex, topIndex, order)

            plt.plot(velocities[botVel:topVel+1], dp_table[0])
            plt.title("Total Minkowski" + str(order) +" Order Cost diagram with a window size of " +str(width) +" raw")
            plt.xlabel("Starting velocity of the trace (m/s)")
            plt.ylabel("Minimum value of the sum (|p_i - p_{i-1}|^" + str(order) + ") along the path")
            plt.show()

            print("Here is a graph of the signal trace across time")       

            plt.figure(figsize=(10, 6))
            myplot = SpectrogramObject.plot()
            plt.xlim((SpectrogramObject.time[startTime], SpectrogramObject.time[stopTime]))

            myplot.plot(time[startTime: stopTime+1], velocities[signal], 'b-', alpha = 0.4, label="Minimum Cost")
            if vStartInd != None:
                # Compute the signal seam for assuming this is the start point.
                seam = reconstruction(p_table, vStartInd-botVel, botVel)
                myplot.plot(time[startTime: stopTime+1], velocities[seam], 'r--', alpha = 0.4, label="Expected Start Point")
            myplot.title("Velocity as a function of time for the minimum cost seam with Minkowski" + str(order)+ " and a window size of " + str(width) + " raw")
            myplot.legend()
            myplot.show()



def documentationMain():
    # Set up

    MySpect, sTime, eTime, signalJump, botInd, topVelInd, visSigIndex = setupForDocumentation()

    # Now document.
    mainTest(MySpect, sTime, eTime, signalJump, botInd, topVelInd, visSigIndex)


def setupForDocumentation():
    # Set up
    filename = "../See_if_this_is_enough_to_get_started/CH_4_009.dig"
    t1 = 14.2389/1e6
    t2 = 31.796/1e6
    MySpect = Spectrogram(filename)
    sTime = MySpect._time_to_index(t1)
    eTime = MySpect._time_to_index(t2)

    bottomVel = 1906.38
    botInd = MySpect._velocity_to_index(bottomVel)

    topVel = 5000
    topVelInd = MySpect._velocity_to_index(topVel)

    signalJump = 25

    visualSignalStart = 2652.21
    visSigIndex = MySpect._velocity_to_index(visualSignalStart)

    return MySpect, sTime, eTime, signalJump, botInd, topVelInd, visSigIndex