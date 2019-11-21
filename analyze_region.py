# coding:utf-8

"""
::

  Author:  LANL Clinic 2019 --<lanl19@cs.hmc.edu>
  Purpose: Analyze a rectangular region of a spectrogram
  Created: 11/20/19
"""
from spectrogram import Spectrogram
import numpy as np
import pandas as pd
from gaussian import Gaussian
import matplotlib.pyplot as plt

def analyze_region(spectrogram:Spectrogram, roi:dict):
  """
  
  """
  time, velocity, intensity = spectrogram.slice(
    roi['time'], roi['velocity']
  )
  # Do we want to convert to power?
  power = spectrogram.power(intensity)
  # let's look at the peak power at each time
  peaks = np.max(power, axis=0)
  speaks = sorted(peaks)
  threshold = speaks[int(len(peaks)*0.04)]
  amax = np.argmax(power, axis=0)
  power[power<threshold] = 0.0
  
  span = 30
  times, centers, widths, amps = [],[],[],[]
  def peg(x):
    if x < 0:
      return 0
    if x >= len(peaks):
      x = len(peaks) - 1
    return x
  for t in range(len(time)-1):
    if power[amax[t],t] > threshold:
      acenter = amax[t]
      vfrom, vto = peg(acenter-20), peg(acenter+20)
      gus = Gaussian(velocity[vfrom:vto], power[vfrom:vto, t])
      if gus.valid:
        times.append(time[t])
        centers.append(gus.center)
        widths.append(gus.width)
        amps.append(gus.amplitude)
    
  
  
  #power[power>=threshold] = 1
  plt.pcolormesh(time*1e6, velocity, power)
  plt.plot(np.array(times)*1e6, centers, 'r.', alpha=0.5)
  plt.show()


if __name__ == "__main__":
  sp = Spectrogram('../dig/sample')
  roi = dict(time=(13.6636e-6,35.1402e-6), velocity=(2393.33,4500.377))
  analyze_region(sp, roi)
  print("Hi!")
  