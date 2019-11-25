import os

from spectrogram import Spectrogram


def generate_graphs(directory):
    for file in os.listdir(directory):
        print(directory + file)
        if file.endswith(".dig"):
            # Then we have the right file.

            Spectrogram_Object = Spectrogram(directory + file)
            sampleSize = 256
            full_length_spectra = Spectrogram_Object.spectrogram(
                0, Spectrogram_Object.samples, sampleSize)

            colormaps = ["gist_stern", "tab20c", "tab20b", "terrain"]

            for colormap in colormaps:
                Spectrogram_Object.plot(
                    full_length_spectra,
                    sample_window=sampleSize,
                    cmap=colormap)
