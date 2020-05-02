#Gallerymaker.py

from spectrogram import Spectrogram

import matplotlib.pyplot as plt
import numpy as np
import os
# import pipelineProcessor as process
from peak_follower import PeakFollower
from jsonDriver import JsonReadDriver
from jsonDriver import JsonWriteDriver
import datetime
import cv2

# process command line args
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--file_name',type=str,default=False)
parser.add_argument('--peak_follow', type = str,default = True)
parser.add_argument('--manual_start', type = bool, default = True)
parser.add_argument('-json_name',type = str, default = "CH_1.json")
parser.add_argument('--velocity_cutoff', type = int, default = 10000)
parser.add_argument('--denoise', type=bool,default = False)
parser.add_argument('--colormap', type=str, default = 'viridis')
# parser.add_argument('--process_name',type=)
args = parser.parse_args()


#Just set directory here, I can't be bothered
directory = '/home/max/clinic/dig/pipeline/'
saveTo = '/home/max/clinic/dig/digImages/'

a = datetime.datetime.now()
time = ('%02d.%02d %02d:%02d'%(a.month,a.day,a.hour,a.minute))
try:
    os.mkdir(saveTo + time + '/')
except FileExistsError:
    print("Oooops! you tried again to fast, storing in same directory")
newDirectory = saveTo + time + '/'


jsondata = JsonReadDriver(args.json_name)

#Create the dictionary to store the succesful runs
jsonwriting = JsonWriteDriver(newDirectory+time)

for filename in os.listdir(directory):
    if filename.endswith(".dig"):
        
        spec = Spectrogram(directory+filename)
        # print(args.JsonRead_name)
        t,v,i = spec.slice((spec.t_start,spec.t_end),(0,args.velocity_cutoff))
        

        if args.manual_start:
            start_coords = jsondata.getManualStart(filename)



        if (args.denoise == True):
            max = np.amax(i)
            # print(max)
            data = i/max
            data = data * 255
            # print(data)
            data = data.astype(np.uint8)  # normalize the data to 0 - 1
            out = cv2.fastNlMeansDenoising(data,None,50,7,21)
            # print('out-',out)
            i=out

        if (args.peak_follow == True):
            print('running',args.peak_follow)
            peaks = PeakFollower(spec,start_coords)
            peaks.run()
            # tsec, v = peaks.v_of_t
            # print(peaks.results)        
            
            
            trace_v = np.array(peaks.results['velocities'])
            trace_t = np.array(peaks.results['times'])
            jsonwriting.store_time_length(filename,trace_t.size)
            plt.plot(trace_t*1e6,trace_v,color = "red")






        #plotting the results

        plt.pcolormesh(t*1e6,v,i,cmap=args.colormap)
        
        plt.savefig(newDirectory + filename+'.png')
        plt.clf()


        continue
    else:
        continue

jsonwriting.flush()
print("done")
