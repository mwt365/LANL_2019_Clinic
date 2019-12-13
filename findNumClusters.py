from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
import EnlargeLabels
import matplotlib.pyplot as plt

def findNumClusters(intensities, minClusters, maxClusters):
    num_clusters = np.arange(minClusters, maxClusters, 1)

    inertia_array = np.zeros((len(num_clusters),))
    width, height = intensities.shape
    flattenedIntensity = intensities.flatten()
    score_array = np.zeros((len(num_clusters),))
    clusteredData = np.zeros((len(num_clusters), len(flattenedIntensity), 1))
    for k_indx in range(len(num_clusters)):
        print("I am on iteration", k_indx+1,"of", len(num_clusters))
        kmeans = KMeans(n_clusters = num_clusters[k_indx], max_iter = 1000, n_jobs = -2)
        outputClusterIndices = kmeans.fit_predict(flattenedIntensity.reshape(-1, 1)) 
        # just cluster by the intensity value.

        inertia_array[k_indx] = kmeans.inertia_
        clusteredData[k_indx] = kmeans.cluster_centers_[outputClusterIndices]

    plt.plot(num_clusters, inertia_array)
    
    EnlargeLabels.increaseReadability("Inertia_ vs num clusters whole dataset","Sum of the distance to the closest cluster center","Num clusters (k)", 25, 20, 20, 16)

    return clusteredData, width, height

from spectrogram import Spectrogram

from plotter import COLORMAPS

def testPlotting(SpectrogramObject:Spectrogram, dataToTest, originalWidth:int, originalHeight:int, timeRange:tuple, velRange:tuple, minClusters:int):

    # dataToTest will be a three dimensional array. The first will be the number of clusters used.
    # the second will be the sample number this will need to reshaped into the appropriate spectrogram style.
    # This way it can be easily converted into the intensity distribution.
    # The third dimension is one long and holds the intensity value for that sample.

    orignalData = SpectrogramObject.intensity
    oldTime = SpectrogramObject.time
    oldVel = SpectrogramObject.velocity

    time, vel, intensity = SpectrogramObject.slice(timeRange, velRange)
    print(time.shape)
    print(vel.shape)
    print(intensity.shape)
    print(dataToTest[0].shape[0]/time.shape[0])
    # time = time[:-1]
    # vel = vel[:-1]
    
    SpectrogramObject.time = time
    SpectrogramObject.velocity = vel

    titleSize = 25
    xlabelSize = 20
    tickMarkSize = 16
    ylab = "Velocity (m/s)"
    xlab = "Time ($\mu$s)"

    for k_indx in range(dataToTest.shape[0]):
        newData = dataToTest[k_indx].reshape(originalWidth, originalHeight)
        print(newData.shape, "shape of new data")
        SpectrogramObject.intensity = newData
        SpectrogramObject.plot()#cmap = COLORMAPS["3w_gby"])
        title = "GEN3CH4_009 Dataset from 0 to 50 $\mu$s and "+str(k_indx+minClusters)+" clusters"
        EnlargeLabels.increaseReadability(title, ylab, xlab, titleSize, xlabelSize, xlabelSize, tickMarkSize)

    # Clean up after the mess.

    SpectrogramObject.intensity = orignalData 
    SpectrogramObject.time = oldTime 
    SpectrogramObject.vel = oldVel