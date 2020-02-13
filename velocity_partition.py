# Vertical Partitioning.



import matplotlib.pyplot as plt
import numpy as np

def make_total_inten_vs_time_plot(Intensity, time, title:str):

    # powered = SpectrogramObject.power(SpectrogramObject.intensity)
    totalInten = np.sum(Intensity, axis=0)
    # powerInten = np.sum(powered, axis=0)


    plt.plot(time*1e6, totalInten)
    plt.title(title.replace("_", "\\_"))
    plt.xlabel('Time ($\mu$s)')
    plt.ylabel("Total Intensity across all velocities")

if __name__ == "__main__":
    import os
    from ProcessingAlgorithms.preprocess.digfile import DigFile
    from spectrogram import Spectrogram
    # plt.clf()
    digFolder = DigFile.dig_dir()
    allDigs = DigFile.inventory()["file"] # Just the files that are not segments.

    saveLoc = os.path.join(os.path.split(digFolder)[0], "PowerIntensityMaps\\")
    if not os.path.exists(saveLoc):
        os.makedirs(saveLoc)

    for i in range(len(allDigs)):
        filename = allDigs[i]
        path = os.path.join(digFolder, filename)
        spec = Spectrogram(path)
        title = spec.data.filename.split('/')[-1]
        inTen = spec.power(spec.intensity)
        make_total_inten_vs_time_plot(inTen, spec.time, title)
        name, _ = os.path.splitext(filename)
        name += "_Power_intensity_map.png"
        name = os.path.join(saveLoc, name.replace("new\\", ""))
        plt.savefig(name)
        plt.clf()
        del spec