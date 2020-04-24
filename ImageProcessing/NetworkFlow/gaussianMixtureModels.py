from sklearn import mixture
import numpy as np

class GMM_Image_Seg:

    def __init__(self, intensityMatrix, **kwargs):
        """
            Input:
                intensityMatrix: (w,h) np array of the intensities in the system.

            Output:
                A 2-component Gaussian Mixture Model that has been fit to the
                input training data using the 2-means centroids as starting
                points for the means of the Gaussians.
        """
        maxIterations = 500
        clf = mixture.GaussianMixture(n_components=2, max_iter=maxIterations, **kwargs)

        if "verbose" in kwargs:
            print("The classifier has been made")

        clf.fit(intensityMatrix.reshape((-1, 1))) # The (-1, 1) reshapes it to have
        # only samples with one feature (pixel intensity).

        if "verbose" in kwargs:
            print("The classifier has been trained.")
            print("It took", clf.n_iter_, "steps out of a maximum", maxIterations)

        self.clf = clf # It has been fit now.
        

    def computeProbabilities(self, intensityMatrix):
        """
            Input:
                - intensityMatrix: (w,h) np array of the intensities in the system.
            Output:
                - array of floats corresponding to the probability 
                that the input intensity is came from each component 
                of the model.
        """

        return self.clf.predict_proba(intensityMatrix.reshape(-1,1))

