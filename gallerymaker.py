#Gallerymaker.py

from spectrogram import Spectrogram

import matplotlib.pyplot as plt
import numpy as np
import os

# process command line args
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--file_name',type=str,default=False)
args = parser.parse_args()

directory = '../dig'


if args.file_name:
    spec = Spectrogram('../dig/'+args.file_name)
    spec.plot()
    plt.savefig(args.file_name+'.png')
    plt.clf()
    del spec

else:

	for filename in os.listdir(directory):
	    if filename.endswith(".dig"):
	        print(filename)
	        spec = Spectrogram('../dig/'+filename)
	        spec.plot()
	        plt.savefig(filename+'.png')
	        plt.clf()
	        del spec

	        continue
	    else:
	    	continue