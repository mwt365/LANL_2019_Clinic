import networkx as nx
import numpy as np
from spectrogram import Spectrogram
# Potential Graph algorithms that can be used to solve the network flow problem.
from networkx.algorithms.flow import *

from ImageProcessing.NetworkFlow.gaussianMixtureModels import GMM_Image_Seg

def setUpGraph(SpectrogramObject:Spectrogram):
    G = nx.DiGraph()

    t = SpectrogramObject.time
    velocity = SpectrogramObject.velocity
    Intensity = SpectrogramObject.intensity
    sigma = 30

    GMM = GMM_Image_Seg(Intensity)

    probabilities = GMM.computeProbabilities(Intensity)

    foregroundLikelihood = probabilities[:,0]/np.sum(probabilities, axis=1)

    count = 0
    lent = len(t)
    lenv = len(velocity)
    maxCount = lent*lenv

    for velInd in range(lenv):
        for timeInd in range(lent):
            u = velInd*lent + timeInd
            inten1 = Intensity[velInd][timeInd]
            
            if velInd + 1 < len(velocity):
                cap1 = penaltyFunction(inten1, Intensity[velInd+1][timeInd], sigma)
                G.add_edge(u, u+lent, capacity = cap1)
            if velInd - 1 >= 0:
                cap2 = penaltyFunction(inten1, Intensity[velInd-1][timeInd], sigma)
                G.add_edge(u, u-lent, capacity = cap2)
            if timeInd + 1 < len(t):
                cap3 = penaltyFunction(inten1, Intensity[velInd][timeInd+1], sigma)
                G.add_edge(u, u+1, capacity = cap3)
            if timeInd - 1 >= 0:
                cap4 = penaltyFunction(inten1, Intensity[velInd][timeInd-1], sigma)
                G.add_edge(u, u-1, capacity = cap4)

            probabilities = GMM.computeProbabilities(inten1)

            capacityFromS = foregroundLikelihood[u]
            capacityToT = 1 - capacityFromS

            G.add_edge("s", u, capacity = capacityFromS)            
            G.add_edge(u, "t", capacity = capacityToT)
            count += 1
            if count%1e4 == 0:
                print("I have setup", count, "nodes out of", maxCount, "nodes.")

    return G

def penaltyFunction(Intensity1, Intensity2, sigma):
    return np.exp(-(np.square(Intensity1-Intensity2)/(2*np.square(sigma))))


def segmentImage(SpectrogramObject:Spectrogram):
    """
        Input:
            - SpectrogramObject: A spectrogram object to be segmented into 
            foreground and background. 

        Output:
            numpy array of pixels indices that are in the foreground (signal)     
    """
    print("The graph will now be set up.")
    G = setUpGraph(SpectrogramObject)

    cutVal, partition = minimum_cut(G, "s", "t")

    print("This is the value of the minimum cost s-t cut", cutVal)

    reach_Foreground, nonreach_Background = partition

    print("This is the number of Foreground States", len(reach_Foreground))
    # print("These are the states in the Foreground", list(reach_Foreground))
    print("This is the number of Background States", len(nonreach_Background))

    # reach_Foreground = reach_Foreground.remove("s") # Do not need the vertex s.

    print("Everyone Has the edges from s and to t, but floating capacities")

    reach = [x for x in reach_Foreground if x != "s"]
    highlightT = []
    highV = []
    lent = len(SpectrogramObject.time)
    # Convert back to indices in time and velocity.
    for state in reach:
        velInd = state//lent
        timeInd = state - velInd*lent
        highlightT.append(timeInd)
        highV.append(velInd)
    
    return np.array(highlightT), np.array(highV)