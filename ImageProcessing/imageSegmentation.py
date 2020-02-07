import networkx as nx
import numpy as np
# Potential Graph algorithms that can be used to solve the network flow problem.
from networkx.algorithms.flow import *

from NetworkFlow.gaussianMixtureModels import GMM_Image_Seg

def setUpGraph(Intensity: np.array, lenVelocity: int, lenTime: int, sigma: int = 30):
    G = nx.DiGraph()

    GMM = GMM_Image_Seg(Intensity)

    probabilities = GMM.computeProbabilities(Intensity)

    foregroundLikelihood = probabilities[:,0]/np.sum(probabilities, axis=1)

    count = 0

    maxCount = lenTime*lenVelocity

    for velInd in range(lenVelocity):
        for timeInd in range(lenTime):
            u = velInd*lenTime + timeInd
            inten1 = Intensity[velInd][timeInd]
            
            if velInd + 1 < lenVelocity:
                cap1 = penaltyFunction(inten1, Intensity[velInd+1][timeInd], sigma)
                G.add_edge(u, u+lenTime, capacity = cap1)
            if velInd - 1 >= 0:
                cap2 = penaltyFunction(inten1, Intensity[velInd-1][timeInd], sigma)
                G.add_edge(u, u-lenTime, capacity = cap2)
            if timeInd + 1 < lenTime:
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


def segmentImage(Intensity: np.array, lenVelocity: int, lenTime: int, sigma: int = 30):
    """
        Input:
            - Intensity: 2D np array. Shape of (v, t). 
            - lenVelocity: integer v.
            - lenTime: integer t. 
            - sigma: integer defaults to 30. 
                This represents the standard deviation of the Gaussian penalty function.
        Output:
            numpy array of pixels indices that are in the foreground (signal)     
    """
    if Intensity.shape != tuple([lenVelocity, lenTime]):
        raise ValueError("The intensity matrix is expected to have the shape (v, t)")
    print("The graph will now be set up.")
    G = setUpGraph(Intensity, lenVelocity, lenTime)

    cutVal, partition = minimum_cut(G, "s", "t", flow_func= edmonds_karp)

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
    # Convert back to indices in time and velocity.
    for state in reach:
        velInd = state//lenTime
        timeInd = state - velInd*lenTime
        highlightT.append(timeInd)
        highV.append(velInd)
    
    return np.array(highlightT), np.array(highV)