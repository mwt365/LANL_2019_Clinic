from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np

import matplotlib.pyplot as plt

def findNumClusters(intensities, minClusters, maxClusters):
    num_clusters = np.arange(minClusters, maxClusters, 1)

    inertia_array = np.zeros((len(num_clusters),))

    score_array = np.zeros((len(num_clusters),))
    for k_indx in range(len(num_clusters)):
        kmeans = KMeans(n_clusters = num_clusters[k_indx], max_iter = 1000, n_jobs = -2)
        kmeans.fit_predict(intensities)
        inertia_array[k_indx] = kmeans.inertia_


    plt.plot(num_clusters, inertia_array)
    plt.title("Inertia_ vs num clusters whole dataset")
    plt.ylabel("Sum of the distance to the closest cluster center")
    plt.xlabel("k")
    plt.show()

    