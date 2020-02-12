# Vertical Partitioning.

from spectrogram import Spectrogram

import matplotlib.pyplot as plt
import numpy as np

def make_total_inten_vs_time_plot(SpectrogramObject:Spectrogram):

    # powered = SpectrogramObject.power(SpectrogramObject.intensity)
    print("My shape", SpectrogramObject.intensity.shape)
    totalInten = np.sum(SpectrogramObject.intensity, axis=0)
    # powerInten = np.sum(powered, axis=0)


    plt.plot(SpectrogramObject.time*1e6, totalInten)
    title = SpectrogramObject.data.filename.split('/')[-1]
    plt.title(title.replace("_", "\\_"))
    plt.xlabel('Time ($\mu$s)')
    plt.ylabel("Total Intensity across all velocities")

if __name__ == "__main__":
    import os
    from ProcessingAlgorithms.preprocess.digfile import DigFile
    # plt.clf()
    digFolder = DigFile.dig_dir()
    allDigs = DigFile.inventory()["file"] # Just the files that are not segments.

    saveLoc = os.path.join(os.path.split(digFolder)[0], "TotalIntensityMaps\\")
    if not os.path.exists(saveLoc):
        os.makedirs(saveLoc)

    for i in range(len(allDigs)):
        filename = allDigs[i]
        path = os.path.join(digFolder, filename)
        spec = Spectrogram(path)
        make_total_inten_vs_time_plot(spec)
        name, _ = os.path.splitext(filename)
        name += "_total_intensity_map.png"
        name = os.path.join(saveLoc, name.replace("new\\", ""))
        plt.savefig(name)
        plt.clf()
        del spec