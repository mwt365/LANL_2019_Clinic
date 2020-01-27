import networkx as nx
import numpy as np
from spectrogram import Spectrogram
# Potential Graph algorithms that can be used to solve the network flow problem.
from networkx.algorithms.flow import *

if False:
    G = nx.DiGraph()
    G.add_edge('x',tuple(['v1', 'a']), capacity=3.0)
    G.add_edge('x',tuple(['v2', 'b']), capacity=1.0)
    G.add_edge(tuple(['v1', 'a']),tuple(['v3', 'c']), capacity=3.0)
    G.add_edge(tuple(['v2', 'b']),tuple(['v3', 'c']), capacity=5.0)
    G.add_edge(tuple(['v2', 'b']),tuple(['v4', 'd']), capacity=4.0)
    G.add_edge(tuple(['v4', 'd']),tuple(['v5', 'e']), capacity=2.0)
    G.add_edge(tuple(['v3', 'c']),'y', capacity=2.0)
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

    count = 0
    lent = len(t)
    lenv = len(velocity)
    maxCount = lent*lenv

    for velInd in range(lenv):
        for timeInd in range(lent):
            u = velInd*lent + timeInd
            inten1 = Intensity[velInd][timeInd]
            cap1 = None
            cap2 = None
            cap3 = None
            cap4 = None
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
            
            incap = np.nanmax(np.array([cap1, cap2, cap3, cap4], dtype=np.float64))
            G.add_edge("s", u, capacity = incap)
            G.add_edge(u, "t", capacity = incap)
            count += 1
            if count%1e6 == 0:
                print("I have completed", count, "out of", maxCount)
    return G

def penaltyFunction(Intensity1, Intensity2, sigma):
    return int(100*np.exp(-(np.square(Intensity1-Intensity2)/(2*np.square(sigma)))))

def foregroundLikelihood(SpectrogramObject:Spectrogram, pixel:int, tLen:int):
    velInd = pixel//tLen
    timeInd = pixel - velInd*tLen

    maxValue = np.max(SpectrogramObject.intensity[velInd])
    return SpectrogramObject.intensity[velInd][timeInd]/maxValue

def backgroundLikelihood(SpectrogramObject:Spectrogram, pixel:int, tLen:int):
    return 1 - foregroundLikelihood(SpectrogramObject, pixel, tLen)

def segmentImage(SpectrogramObject:Spectrogram):
    G = setUpGraph(SpectrogramObject)

    cutVal, partition = minimum_cut(G, "s", "t")

    print("This is the value of the minimum cost s-t cut", cutVal)

    reach_Foreground, nonreach_Background = partition

    print("This is the number of Foreground States", len(reach_Foreground))
    print("These are the states in the Foreground", list(reach_Foreground))
    print("This is the number of Background States", len(nonreach_Background))

    highlighted = np.zeros(SpectrogramObject.intensity.shape, dtype=bool)
    lent = len(SpectrogramObject.time)

    count = 0
    for state in reach_Foreground:
        if state != "s":
            velInd = state//lent
            timeInd = state - velInd*lent

            highlighted[velInd][timeInd] = 1
            count += 1

    
    
    SpectrogramObject.intensity = highlighted
    print("This is the number of pixels that are set to 1", count)

    SpectrogramObject.plot()
