# Cutting both sides
import numpy as np
import matplotlib.pyplot as plt

def cutAndPlot(vel, intensity, upperVel, lowerVel):
    if upperVel < lowerVel:
        # just swap them.
        tmp = lowerVel
        lowerVel = upperVel
        upperVel = tmp
    cutTop = vel[np.where(vel <= upperVel)]
    cutBoth = cutTop[np.where(cutTop >= lowerVel)]

    cutTopIntensity = intensity[np.where(vel <= upperVel)]
    cutBothIntensity = cutTopIntensity[np.where(cutTop >= lowerVel)]


    plt.plot(cutBoth, cutBothIntensity)
    plt.xlabel("Velocity (m/s)")
    plt.ylabel("Power")

    plt.show()