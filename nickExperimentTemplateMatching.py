# coding:utf-8

"""
::

  Author:  LANL Clinic 2019 --<lanl19@cs.hmc.edu>
  Purpose: Experiment to determine which Templates work best.
  Created: 04/10/20
"""


from spectrogram import Spectrogram
from ImageProcessing.Templates.templates import Templates
from template_matcher import TemplateMatcher

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import tqdm # For printing a timing information to see how far you have gotten.

experimentDataFileName = r"..\NickEstimateOfJumpOffPosition.xlsx"

data = pd.read_excel(experimentDataFileName)

columns = data.columns

data = data.dropna()
data.reset_index()

files = data["Filename"].to_list()
bestGuessTimes = (data["starting time (bottom signal if frequency multiplexed)"]*1e6).to_list()
bestGuessVels = data["starting velocity (bottom signal if frequency multiplexed)"].to_list()

baseFolder = r"..\dig"

saveFolder = r"..\templateMatchingExperiment"
if not os.path.exists(saveFolder):
    os.makedirs(saveFolder)

methods = ['cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
keyedMethods = ['TM_SQDIFF', 'TM_SQDIFF_NORMED']
output = [[] for i in range(len(Templates))]
zeros = np.zeros(len(files))


# Set up the data frames.
print("Setting up the experiment data locations.")

for ind in range(len(Templates)):
    d = {"Filename": files}
    for k in range(len(keyedMethods)):
        d[f"{keyedMethods[k]} Time microseconds"] = zeros
        d[f"{keyedMethods[k]} Velocity (m/s)"] = zeros
        d[f"{keyedMethods[k]} L2 error"] = zeros
        d[f"{keyedMethods[k]} L1 error"] = zeros
        
    output[ind] = d

outputFiles = [[] for i in range(len(files))]
zeros = np.zeros(len(Templates))
for ind in range(len(files)):
    d = {"Template": Templates}
    for k in range(len(keyedMethods)):
        d[f"{keyedMethods[k]} Time microseconds"] = zeros
        d[f"{keyedMethods[k]} Velocity (m/s)"] = zeros
        d[f"{keyedMethods[k]} L2 error"] = zeros
        d[f"{keyedMethods[k]} L1 error"] = zeros
        
    outputFiles[ind] = d
print("The experiment is now set up")
# Run the experiment.

for i in tqdm.trange(len(files)):
    filename = files[i]
    fname = os.path.join(baseFolder, filename)
    MySpect = Spectrogram(fname, overlap=7/8)

    bestGuessTime = bestGuessTimes[i]
    bestGuessVel = bestGuessVels[i]

    template_matcher = TemplateMatcher(MySpect, None, Templates.start_pattern.value, span = 200, k=1)

    valuesList = template_matcher.matchMultipleTemplates(methods, Templates)
    for ind in range(len(valuesList)):
        times, velos, scores, methodUsed = valuesList[ind]

        for k in range(len(times)):
            output[ind][keyedMethods[methodUsed[k]] + " Time microseconds"][i] = times[k]
            output[ind][keyedMethods[methodUsed[k]] + " Velocity (m/s)"][i] = velos[k]
            output[ind][keyedMethods[methodUsed[k]] + " L2 error"][i] = np.sqrt((times[k] - bestGuessTime)**2 + (velos[k] - bestGuessVel)**2)
            output[ind][keyedMethods[methodUsed[k]] + " L1 error"][i] = np.abs(times[k] - bestGuessTime) + np.abs(velos[k] - bestGuessVel)
    

            # Save it in the file format.
            outputFiles[i][keyedMethods[methodUsed[k]] + " Time microseconds"][ind] = times[k]
            outputFiles[i][keyedMethods[methodUsed[k]] + " Velocity (m/s)"][ind] = velos[k]
            outputFiles[i][keyedMethods[methodUsed[k]] + " L2 error"][ind] = np.sqrt((times[k] - bestGuessTime)**2 + (velos[k] - bestGuessVel)**2)
            outputFiles[i][keyedMethods[methodUsed[k]] + " L1 error"][ind] = np.abs(times[k] - bestGuessTime) + np.abs(velos[k] - bestGuessVel)

    del MySpect

for ind, template in tqdm.tqdm(enumerate(Templates)):
    saving = pd.DataFrame(output[ind])
    saving.to_csv(os.path.join(saveFolder, str(template) + ".csv"))

print("saved all the files in the original format")

for i in tqdm.trange(len(files)):
    if not os.path.exists(os.path.join(saveFolder, files[i].replace(".dig", ""))):
        os.makedirs(os.path.join(saveFolder, files[i].replace(".dig", "")))
    d = pd.DataFrame(outputFiles[i])
    d.to_csv(os.path.join(saveFolder, files[i].replace(".dig", "") + "_templateMatching.csv"))

print("Experiment completed.")