#greedy.py

from spectrogram import *

FILENAME = 'GEN3CH_4_009.dig'

spec = Spectrogram(FILENAME)

print('filestart = ', spec.t_start)
print('fileend = ', spec.t_end)

sliced = spec.slice((spec.t_start,spec.t_start+specdata.dt),(0,100))