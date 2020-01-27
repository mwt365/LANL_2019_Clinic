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


def setUpGraph(SpectrogramObject:Spectrogram, foregroundPoints:list, backgroundPoints:list):
    G = nx.DiGraph()

    t = SpectrogramObject.time
    velocity = SpectrogramObject.velocity
    Intensity = SpectrogramObject.intensity
    sigma = 30

    hyperLambda = 0.5

    count = 0
    lent = len(t)
    lenv = len(velocity)
    maxCount = lent*lenv

    foreGroundCost = foregroundLikelihood(SpectrogramObject, foregroundPoints, sigma)
    backGroundCost = foregroundLikelihood(SpectrogramObject, backgroundPoints, sigma)


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
            capacityFromS = 0
            capacityToT = backGroundCost
            if (timeInd, velInd) not in foregroundPoints and (timeInd, velInd) not in backgroundPoints:
                capacityFromS = hyperLambda * Rvalue(inten1, backgroundPoints)
                capacityToT = hyperLambda * Rvalue(inten1, foregroundPoints)
            elif (timeInd, velInd) in foregroundPoints:
                capacityFromS = foreGroundCost
                capacityToT = 0
            
            G.add_edge("s", u, capacity = capacityFromS)            
            G.add_edge(u, "t", capacity = capacityToT)
            count += 1
            if count%1e6 == 0:
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

def foregroundLikelihood(SpectrogramObject:Spectrogram, foregroundPoints:list, sigma:int):
    
    tLen = len(SpectrogramObject.time)
    vLen = len(SpectrogramObject.velocity)
    Intensity = SpectrogramObject.intensity
    maxPenalty = 0
    for (timeInd, velInd) in foregroundPoints:
        inten1 = Intensity[velInd][timeInd]
        if velInd + 1 < vLen:
            cap1 = penaltyFunction(inten1, Intensity[velInd+1][timeInd], sigma)
            maxPenalty = max(maxPenalty, cap1)
        if velInd - 1 >= 0:
            cap2 = penaltyFunction(inten1, Intensity[velInd-1][timeInd], sigma)
            maxPenalty = max(maxPenalty, cap2)
        if timeInd + 1 < tLen:
            cap3 = penaltyFunction(inten1, Intensity[velInd][timeInd+1], sigma)
            maxPenalty = max(maxPenalty, cap3)
        if timeInd - 1 >= 0:
            cap4 = penaltyFunction(inten1, Intensity[velInd][timeInd-1], sigma)
            maxPenalty = max(maxPenalty, cap4)
    return 1 + maxPenalty

# def backgroundLikelihood(SpectrogramObject:Spectrogram, pixel:int, tLen:int):
#     return 1 - foregroundLikelihood(SpectrogramObject, pixel, tLen)


def Rvalue(pixelIntensity:float, intensityList:list):
    """
    Based upon this paper:
        https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=17&cad=rja&uact=8&ved=2ahUKEwjek92Qk6PnAhXFPn0KHZeaDf44ChAWMAZ6BAgHEAE&url=http%3A%2F%2Fyadda.icm.edu.pl%2Fyadda%2Felement%2Fbwmeta1.element.baztech-article-AGH1-0028-0094%2Fc%2FFabijanska.pdf&usg=AOvVaw0Xm9HxUAa7BUn6dqAPQ7qI

    """
    mean = np.mean(intensityList)

    return 1 - (pixelIntensity - mean)/mean


def segmentImage(SpectrogramObject:Spectrogram, foregroundPoints:list, backgroundPoints:list):
    """
        foregroundPoints: list of time, velocity coordinates that correspond to the signal.
        backgroundPoints: list of time, velocity coordinates that correspond to the noise.

        Output:
            numpy array of pixels indices that are in the foreground (signal)     
    """
    G = setUpGraph(SpectrogramObject, foregroundPoints, backgroundPoints)

    cutVal, partition = minimum_cut(G, "s", "t")

    print("This is the value of the minimum cost s-t cut", cutVal)

    reach_Foreground, nonreach_Background = partition

    print("This is the number of Foreground States", len(reach_Foreground))
    # print("These are the states in the Foreground", list(reach_Foreground))
    print("This is the number of Background States", len(nonreach_Background))

    # reach_Foreground = reach_Foreground.remove("s") # Do not need the vertex s.

    print("Everyone Has the edges from s and to t, but floating capacities")

    return np.array(reach_Foreground)

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
