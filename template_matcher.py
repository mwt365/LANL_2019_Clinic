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
from template_matching import Template
from spectrogram import Spectrogram
from ImageProcessing.Templates.templates import *
from scipy.misc import imsave
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle


#TODO 'Genetic' algorithm for finding templates
#TODO Run peak follower on highest scored points with contrast, or highest signal score


class TemplateMatcher():
    """
    Tries to find the local maximum from scores returned by various
    OpenCV template matching algorithms, based on user input.

    **Inputs to the constructor**

    - spectrogram: an instance of Spectrogram
    - start_point: (t, v), the coordinates at which to begin the search
    - template: a two-dimensional array that is read into to OpenCV as 
        a template image for matching.
    - span: (60) a 'radius' of pixels surrounding the starting point to 
        increase the searching space for the starting point.

    """

    def __init__(self, spectrogram, start_point, template, span=80):
        assert isinstance(spectrogram, Spectrogram)
        assert (start_point is not None)
        assert (len(template) > 0)

        self.template = template
        self.raw_click = start_point
        self.spectrogram = spectrogram
        self.span = int(span)

        self.setup()



    def setup(self):

        time, velocity = self.raw_click

        self.click = (self.spectrogram._time_to_index(time), self.spectrogram._velocity_to_index(velocity))

        start_time = self.spectrogram._time_to_index(time)
        end_time = self.spectrogram._time_to_index(time) + self.span
        max_time_index = self.spectrogram.intensity.shape[1]

        start_velo = self.spectrogram._velocity_to_index(velocity)
        end_velo = self.spectrogram._velocity_to_index(velocity) + (3*self.span)
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
        threshold_percentile = np.percentile(sortedmatrix, 90)

        new_matrix = np.where(matrix > threshold_percentile, matrix+threshold_percentile, threshold_percentile)
        new_matrix = np.flip(np.flip(new_matrix), axis=1)
        
        spec = new_matrix[(-1 * velo_bounds[1]):(-1 * velo_bounds[0]), time_bounds[0]:time_bounds[1]]

        imsave("./full.png", new_matrix[:])
        imsave("./debug.png", spec[:])

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

        xcoords = []
        ycoords = []
        scores = []

        for meth in methods:
            img = img2.copy()
            method = eval(meth)

            # Apply template Matching
            res = cv2.matchTemplate(img,template,method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            
            # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                top_left = min_loc
                scores.append(min_val)
            else:
                top_left = max_loc
                scores.append(max_val)
            bottom_right = (top_left[0] + w, top_left[1] + h)

            velo_offset = self.spectrogram.velocity[self.velo_bounds[0]]
            time_offset = self.spectrogram.time[self.time_bounds[0]] * 10**6

            xcoords.append( (self.spectrogram.time[top_left[0]] * 10**6) + time_offset)
            ycoords.append( self.spectrogram.velocity[top_left[1]] + velo_offset)

            # cv2.rectangle(img, top_left, bottom_right, 255, thickness=1)
            # plt.subplot(121),plt.imshow(res,cmap = 'gray')
            # plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
            # plt.subplot(122),plt.imshow(img,cmap = 'gray')
            # plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
            # plt.suptitle(meth)
            # plt.show()

        return xcoords, ycoords




# if __name__ == "__main__":

    # path = "/Users/trevorwalker/Desktop/Clinic/For_Candace/newdigs/CH_2_009.dig"
    # spec = Spectrogram(path, 0.0, 60.0e-6, overlap_shift_factor= 1/8, form='db')

    # import random
    # secure_random = random.SystemRandom()

    # template = opencv_start_pattern2

    # time = round(secure_random.uniform(8.5, 13.5), 3)
    # velo = round(secure_random.uniform(2400.5, 2700.5), 3)

    # print("   time : ",time)
    # print("velocity: ",velo,'\n')

    # user_click = (time*1e-6, velo)

    # template_matcher = TemplateMatcher(spec, user_click, template, span=80)
    # times, velos = template_matcher.main()


