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

def threshold(totalInten, fracOfMedian):
    """
        Return the time indices that contain total intensity less than 
        fracOfMedian of the maximum throughout the whole spectrogram.
    """

    maxValue = np.max(totalInten)

    inds = np.where(totalInten <= fracOfMedian*maxValue)

    print("The number of locations that I think are separators is", len(inds[0]), "for a threshold of", fracOfMedian)


    return inds


if __name__ == "__main__":
    import os
    from ProcessingAlgorithms.preprocess.digfile import DigFile
    from spectrogram import Spectrogram
    # plt.clf()
    digFolder = DigFile.dig_dir()
    allDigs = DigFile.inventory()["file"] # Just the files that are not segments.

    saveLoc = os.path.join(os.path.split(digFolder)[0], "TotalIntensityMaps\\Median\\")
    if not os.path.exists(saveLoc):
        os.makedirs(saveLoc)

    # for i in range(len(allDigs)):
    #     filename = allDigs[i]
    #     path = os.path.join(digFolder, filename)
    #     spec = Spectrogram(path)
    #     title = spec.data.filename.split('/')[-1]
    #     inTen = spec.power(spec.intensity)
    #     make_total_inten_vs_time_plot(inTen, spec.time, title)
    #     name, _ = os.path.splitext(filename)
    #     name += "_Power_intensity_map.png"
    #     name = os.path.join(saveLoc, name.replace("new\\", ""))
    #     plt.savefig(name)
    #     plt.clf()
    #     del spec
    
    fractions = np.arange(10)
    fractionsL = len(fractions)

    data = np.zeros((fractionsL, len(allDigs)))
    data[:,0] = fractions
    for i in range(len(allDigs)):
        filename = allDigs[i]
        path = os.path.join(digFolder, filename)
        spec = Spectrogram(path)
        inTen = np.sum(spec.intensity, axis = 0)
            # Offset the values so that everything is non negative.
        inTen = inTen - np.min(inTen)

        for j in range(fractionsL):
            frac = np.power(10.0, -1*fractions[j])
            data[j,i] = threshold(inTen, frac)
        
        del spec

    name = os.path.join(saveLoc, "NumberOfLowValuesVsFracOfMax.txt")

    np.savetxt(name, data)
