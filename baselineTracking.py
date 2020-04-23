import baselines
from spectrogram import Spectrogram

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import tqdm # For printing a timing information to see how far you have gotten.






def saveBaselineIntensityImages(files, saveLoc:str = None, imageExt:str = "png"):
    """
    Save a image file of the baseline as a function of time to the folder specified by saveLoc.
    """
    if saveLoc == None:
        saveLoc = r"../baselineIntensityMaps"
    
    for i in tqdm.trange(len(files)):
        filename = f"../dig/{files[i]}"
        MySpect = Spectrogram(filename)
        peaks, _, heights = baselines.baselines_by_squash(MySpect)

        plt.plot(np.array(MySpect.time)*1e6, MySpect.intensity[MySpect._velocity_to_index(peaks[0])])
        plt.xlabel("Time ($\mu$s)")
        plt.ylabel("Intensity (db)")
        plt.title(f"{MySpect.data.filename}")
        path = os.path.join(saveLoc, files[i].replace(".dig", "") + f"_baselineIntensity.{imageExt}")
        if not os.path.exists(os.path.split(path)[0]):
            os.makedirs(os.path.split(path)[0])

        plt.savefig(path)

        # Reset the state for the next data file.
        plt.clf()
        del MySpect

def baselineTracking(spectrogram, baselineVel, changeThreshold):
    """
        Return the first time in microseconds that baseline's intensity value changes outside the changeThreshold.
        Use the average baseline value as the estimate
    """

    baselineInd = spectrogram._velocity_to_index(baselineVel)
    runningAverage = spectrogram.intensity[baselineInd][0]
    
    jumpOffTime = spectrogram.probe_destruction_time_max

    for ind in range(spectrogram.intensity.shape[-1]):
        currentIntensity = spectrogram.intensity[baselineInd][ind]
        if (np.abs(currentIntensity) >= np.abs((1+changeThreshold) * runningAverage)) or (np.abs(currentIntensity) <= np.abs((1-changeThreshold) * runningAverage)):
            return spectrogram.time[ind]*1e6
        runningAverage += 1/(ind + 1)*(currentIntensity - runningAverage)
    return jumpOffTime


def runExperiment(thresholds, trainingFilePath):

    data = pd.read_excel(trainingFilePath)

    data = data.dropna() # Ignore the samples that we do not have the full data for.
    data.reset_index() # Reset so that it can be easily indexed.

    files = data["Filename"].to_list()
    bestGuessTimes = (data["starting time (bottom signal if frequency multiplexed)"]*1e6).to_list()
    bestGuessVels = data["starting velocity (bottom signal if frequency multiplexed)"].to_list()


    d = {"Filename": files}

    for i in range(len(thresholds)):
        d[f"threshold {thresholds[i]}"] = 0
        d[f"threshold {thresholds[i]} error"] = 0



    for i in tqdm.trange(len(files)):
        filename = f"../dig/{files[i]}"
        MySpect = Spectrogram(filename)
        peaks, _, heights = baselines.baselines_by_squash(MySpect)


        for thres in thresholds:
            
            timeEstimate = baselineTracking(MySpect, peaks[0], thres)

            d[f"threshold {thres}"] = timeEstimate
            d[f"threshold {thres} error"] = bestGuessTimes[i] - timeEstimate

    # Compute summary statistics
    summaryStatistics = np.zeros((len(thresholds), 5))

    for i in tqdm.trange(len(thresholds)):
        summaryStatistics[i][0] = np.mean(d[f"threshold {thresholds[i]} error"]**2)
        summaryStatistics[i][1] = np.std(d[f"threshold {thresholds[i]} error"]**2)
        summaryStatistics[i][2] = np.median(d[f"threshold {thresholds[i]} error"]**2)
        summaryStatistics[i][3] = np.max(d[f"threshold {thresholds[i]} error"]**2)
        summaryStatistics[i][4] = np.min(d[f"threshold {thresholds[i]} error"]**2)

    print(f"The minimum avg L2 error is {np.min(summaryStatistics[:, 0])} which occurs at {np.argmin(summaryStatistics[:, 0])} i.e. threshold of {thresholds[np.argmin(summaryStatistics[:, 0])]}")
    print(f"The statistics for that the threshold is {summaryStatistics[np.argmin(summaryStatistics[:, 0])]}")

    print(f"Saving the raw data to baselineTracking.csv")

    d = pd.DataFrame(d, index=d["Filename"])
    d.to_csv("baselineTracking.csv")

    myMin = str(min(thresholds)).replace(".", "_")
    myMax = str(max(thresholds)).replace(".", "_")
    print(f"Saving the summary statistics to baselineTracking_thres_{myMin}-{myMax}.csv")

    np.to_csv(f"baselineTracking_thres_{myMin}-{myMax}.csv")

if __name__ == "__main__":
    thresholds = np.linspace(0.3, 1)
    saveLoc = r"../baselineIntensityMaps"
    experimentFileName = r"../NickEstimateOfJumpOffPosition.xlsx"

    data = pd.read_excel(experimentFileName)

    data = data.dropna() # Ignore the samples that we do not have the full data for.
    data.reset_index() # Reset so that it can be easily indexed.

    files = data["Filename"].to_list()

    if False:
        saveBaselineIntensityImages(files, imageExt = "jpeg")

    runExperiment(thresholds, experimentFileName)