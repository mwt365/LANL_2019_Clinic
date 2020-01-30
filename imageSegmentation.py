import networkx as nx
import numpy as np
from spectrogram import Spectrogram
# Potential Graph algorithms that can be used to solve the network flow problem.
from networkx.algorithms.flow import *

from ImageProcessing.NetworkFlow.gaussianMixtureModels import GMM_Image_Seg

if False:
    G = nx.DiGraph()
    G.add_edge('x',tuple(['v1', 'a']), capacity=3.0)
    G.add_edge('x',tuple(['v2', 'b']), capacity=1.0)
    G.add_edge(tuple(['v1', 'a']),tuple(['v3', 'c']), capacity=3.0)
    G.add_edge(tuple(['v2', 'b']),tuple(['v3', 'c']), capacity=5.0)
    G.add_edge(tuple(['v2', 'b']),tuple(['v4', 'd']), capacity=4.0)
    G.add_edge(tuple(['v4', 'd']),tuple(['v5', 'e']), capacity=2.0)
    G.add_edge(tuple(['v3', 'c']),'y', capacity=2.2)
    G.add_edge(tuple(['v5', 'e']),'y', capacity=3.0)
    R = edmonds_karp(G, 'x', 'y')
    flow_value = nx.maximum_flow_value(G, 'x', 'y')
    cut, partition = nx.minimum_cut(G, "x", "y")
    print("This is is the partition", partition)

    print("This is the residual graph's adjacency list",list(R.adjacency()))


def setUpGraph(SpectrogramObject:Spectrogram):
    G = nx.DiGraph()

    t = SpectrogramObject.time
    velocity = SpectrogramObject.velocity
    Intensity = SpectrogramObject.intensity
    sigma = 30

    GMM = GMM_Image_Seg(Intensity, verbose=True)

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
                print("I have completed", count, "out of", maxCount)

    # for (timeInd, velInd) in foregroundPoints:
    #     u = velInd*lent + timeInd
    #     G.add_edge("s", u, capacity = foreGroundCost)

    # for (timeInd, velInd) in backgroundPoints:
    #     u = velInd*lent + timeInd
    #     G.add_edge(u, "t", capacity = backGroundCost)

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

    return np.array(list(reach_Foreground))

    # count = 0
    # for state in reach_Foreground:
    #     if state != "s":
    #         velInd = state//lent
    #         timeInd = state - velInd*lent

    #         highlighted[velInd][timeInd] = 1
    #         count += 1

    
    
    # SpectrogramObject.intensity = highlighted
    # print("This is the number of pixels that are set to 1", count)

    # SpectrogramObject.plot()
