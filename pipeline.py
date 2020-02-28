#Gallerymaker.py

from spectrogram import Spectrogram

import matplotlib.pyplot as plt
import numpy as np
import os
import pipelineProcessor as process
from peak_follower import PeakFollower
from jsonDriver import JsonDriver

# process command line args
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--file_name',type=str,default=False)
parser.add_argument('--peak_follow', type = bool,default = True)
parser.add_argument('--manual_start', type = bool, default = True)
parser.add_argument('-json_name',type = str)
# parser.add_argument('--process_name',type=)
args = parser.parse_args()


#Just set directory here, I can't be bothered
directory = '/home/lanl/Documents/dig/new/002/'
saveTo = '/home/lanl/Documents/dig/digImages/'



for filename in os.listdir(directory):
    if filename.endswith(".dig"):
        
        spec = Spectrogram(directory+filename)
        # print(args.json_name)
        data = JsonDriver(args.json_name)

        if args.manual_start:
            start_coords = data.getManualStart(filename)





        peaks = PeakFollower(spec,start_coords)
        peaks.run()
        # tsec, v = peaks.v_of_t
        # print(peaks.results)        
        t,v,i = spec.slice((spec.t_start,spec.t_end),(0,10000))
        trace_v = np.array(peaks.results['velocities'])
        trace_t = np.array(peaks.results['times'])
        plt.pcolormesh(t*1e6,v,i)
        plt.plot(trace_t*1e6,trace_v,color = "red")
        plt.show()

        # process.saveToFile(spec,filename, saveTo)
        # print(filename)
        # spec = Spectrogram(directory+filename)
        # spec.plot()
        # plt.savefig(saveTo + filename+'.png')
        # plt.clf()
        # del spec

        continue
    else:
        continue

# def doProcessing(filename):
# 			print(filename)
# 			spec = Spectrogram(directory+filename)
# 			spec.plot()
# 			plt.savefig(saveTo + filename+'.png')
# 			plt.clf()
# 			del spec
