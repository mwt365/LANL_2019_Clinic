from spectrogram import Spectrogram

import matplotlib.pyplot as plt
import numpy as np
import os


def saveToFile(spectrogram,filename,saveTo):
			print(filename)
			spectrogram.plot()
			plt.savefig(saveTo + filename+'.png')
			plt.clf()
			