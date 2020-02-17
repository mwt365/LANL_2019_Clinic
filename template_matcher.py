#!/usr/bin/env python3
# coding:utf-8
"""
::

  Author:  LANL Clinic 2019 --<lanl19@cs.hmc.edu>
  Purpose: Attempt to match templates in a user specified area.
  Created: 2/16/20
"""
import cv2
import numpy as np
from baselines import baselines_by_squash as bline
from follower import Follower
from moving_average import moving_average
from template_matching import Template
from spectrogram import Spectrogram
from ImageProcessing.Templates.templates import *
from scipy.misc import imsave
from matplotlib import pyplot as plt




class TemplateMatcher():
    """
    A naive follower implementation that uses a local-region
    smoothing and then follows the local maximum.

    **Inputs to the constructor**

    - spectrogram: an instance of Spectrogram
    - start_point: (t, v), the coordinates at which to begin the search
    - span: (60) the number of pixels on either side of the starting value of v
      at which to search for a peak at the next time step.
    - smoothing: (4) the number of points on either side of a given velocity
      in a spectrum to average over to produce a smoothed representation of the
      spectrum prior to searching for a peak.
    - max_hop: (50) the largest change in v from the previous time step to consider
      as a continuation.


    """

    def __init__(self, spectrogram, start_point, template, span=80):
        assert isinstance(spectrogram, Spectrogram)
        peaks, dv, heights = bline(spectrogram)
        self.baselines = np.array(peaks)

        self.template = template
        self.raw_click = start_point
        self.spectrogram = spectrogram
        self.span = span

        self.setup()



    def setup(self):

        t, velocity = self.raw_click
        time = t * 10**-6

        self.click = (self.spectrogram._time_to_index(time), self.spectrogram._time_to_index(velocity))

        start_time = self.spectrogram._time_to_index(time) - int(1.5 * self.span)
        end_time = self.spectrogram._time_to_index(time) + int(1.5 * self.span)
        max_time_index = self.spectrogram.intensity.shape[1]

        start_velo = self.spectrogram._velocity_to_index(velocity) - self.span
        end_velo = self.spectrogram._velocity_to_index(velocity) + self.span
        max_velo_index = self.spectrogram.intensity.shape[0]

        if start_time < 0:
            start_time = 0
        if end_time > max_time_index:
            end_time = max_time_index-1
        if start_velo < 0:
            start_velo = 0
        if end_velo > max_velo_index:
            end_velo = max_velo_index-1

        self.time_bounds = (start_time, end_time) #indices, not actual time/velo values
        self.velo_bounds = (start_velo, end_velo)



    def crop_intensities(self, matrix, time_bounds, velo_bounds):

        sortedmatrix = sorted(matrix.flatten(), reverse=True)
        threshold_percentile = np.percentile(sortedmatrix, 95)

        new_matrix = np.where(matrix > threshold_percentile, matrix+(2*threshold_percentile), threshold_percentile)

        new_matrix = new_matrix[velo_bounds[0]:velo_bounds[1], time_bounds[0]:time_bounds[1]]

        spec = np.flip(np.flip(new_matrix), axis=1)

        return spec


    def main(self):

        matrix = self.spectrogram.intensity

        cropped_spectrogram = self.crop_intensities(matrix, self.time_bounds, self.velo_bounds)

        imsave("./im_template.png", self.template[:])
        imsave("./im_cropped_bg.png", cropped_spectrogram[:])


        img = cv2.imread('./im_cropped_bg.png', 0)
        img2 = img.copy()

        template = cv2.imread('./im_template.png', 0)

        w, h = template.shape[::-1]

        # All the 6 methods for comparison in a list
        methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
                    'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']


        # method = 'cv2.TM_CCORR'

        # img = img2.copy()
        # method1 = eval(method)

        # res = cv2.matchTemplate(img,template,method1)
        # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # if method1 in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        #     top_left = min_loc
        # else:
        #     top_left = max_loc
        # bottom_right = (top_left[0] + w, top_left[1] + h)

        # cv2.rectangle(img,top_left, bottom_right, 255, thickness=1)

        # print(top_left, bottom_right)

        # # plt.subplot(121),plt.imshow(res,cmap = 'gray')
        # # plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        # # plt.subplot(122),plt.imshow(img,cmap = 'gray')
        # # plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        # # plt.suptitle(method)

        # # plt.show()

        # return top_left, bottom_right


        for meth in methods:
            img = img2.copy()
            method = eval(meth)

            # Apply template Matching
            res = cv2.matchTemplate(img,template,method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                top_left = min_loc
            else:
                top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)

            cv2.rectangle(img,top_left, bottom_right, 255, thickness=1)

            plt.subplot(121),plt.imshow(res,cmap = 'gray')
            plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
            plt.subplot(122),plt.imshow(img,cmap = 'gray')
            plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
            plt.suptitle(meth)

            plt.show()




if __name__ == "__main__":

    path = "/Users/trevorwalker/Desktop/Clinic/For_Candace/newdigs/CH_2_009.dig"
    spec = Spectrogram(path, 0.0, 60.0e-6, overlap_shift_factor= 1/8, form='db')

    # user_click = (10.7, 2790.1)

    import random
    secure_random = random.SystemRandom()

    template = opencv_start_pattern2

    for i in range(40, 140, 20):

        time = round(secure_random.uniform(8.5, 13.5), 3)
        velo = round(secure_random.uniform(2600.5, 2900.5), 3)

        print("   time : ",time)
        print("velocity: ",velo,'\n')

        user_click = (time, velo)

        template_matcher = TemplateMatcher(spec, user_click, template, span=i)
        template_matcher.main()


