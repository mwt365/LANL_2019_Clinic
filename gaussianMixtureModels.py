from sklearn import mixture
import numpy as np

from spectrogram import Spectrogram

filename = "../dig/GEN3CH_4_009.dig" 

MySpect = Spectrogram(filename)

tEnd = 4e-5
vTop = 6500

t, v, intensity = MySpect.slice((0, tEnd), (0, vTop))

vLen, tLen = intensity.shape
print("The number of data points is", vLen*tLen)
X_train = np.reshape(intensity, (vLen*tLen,)).reshape((-1,1))

clf = mixture.GaussianMixture(n_components=2, n_init=3)
print("I have constructed the classifier", clf)

clf.fit(X_train)

print("Here are the means associated with the components of the model", clf.means_)
print("Here are the associated covariances", clf.covariances_)

predictionsProbabilities = clf.predict_proba(X_train)