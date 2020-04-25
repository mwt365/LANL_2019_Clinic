# coding:utf-8

"""
::

   Author:  LANL Clinic 2019 --<lanl19@cs.hmc.edu>
   Purpose: Make a gallery of images of spectrograms.
   Created: 3/1/20
"""

import os
currDir = os.getcwd()
os.chdir(os.path.split(__file__)[0]) # Since this file is in one level up from the root and we want act as if we are working from there.


from spectrogram import Spectrogram
from ProcessingAlgorithms.preprocess.digfile import DigFile

os.chdir(currDir)

import matplotlib.pyplot as plt

def createGallery(digsToLookAt:list=None, colormap="blue-orange-div", fileext="jpeg", transformData = False):
	if type(digsToLookAt) == type(None):
		digsToLookAt = DigFile.inventory(justSegments = True)['file']

	digDir = DigFile.dig_dir()
	imageDir, _ = os.path.split(digDir) 
	imageFold = "ImageGallery"
	imageDir = os.path.join(imageDir, imageFold)
	print(digsToLookAt)
	for i in range(len(digsToLookAt)):
		filename = digsToLookAt[i]
		print(filename)
		spec = Spectrogram(os.path.join(digDir,filename))
		pcms, lastAxes = spec.plot(transformData, cmap = colormap)
		fileloc, filename = os.path.split(os.path.splitext(os.path.join(imageDir,filename))[0])
		fileloc = os.path.join(fileloc, filename)

		for key in pcms.keys():
			if "complex" in key:
				continue # There is not a graph with a name that contains complex.
			if not os.path.exists(fileloc):
				os.makedirs(fileloc)

			plt.figure(num=key) # Get to the appropriate figure.
			plt.savefig(os.path.join(fileloc,filename) + f' {key} spectrogram.{fileext}')
			plt.clf()
		del spec

if __name__ == "__main__":		
	# process command line args
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--file_name',type=str,default=False)
	args = parser.parse_args()


	#Just set directory here, I can't be bothered
	directory = '/home/lanl/Documents/dig/new/CH_1_009/'

	plt.rcParams["figure.figsize"] = [20, 10] # This is setting the default parameters to "20 by 10" inches by inches.
	# I would prefer if this was set to full screen.

	if args.file_name:
		spec = Spectrogram(directory+args.file_name)
		spec.plot()
		plt.savefig(args.file_name+'.png')
		plt.clf()
		del spec