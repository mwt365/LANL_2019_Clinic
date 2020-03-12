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
from baselines import baselines_by_squash
from template_matching import Template
from spectrogram import Spectrogram
from ImageProcessing.Templates.templates import *
from scipy.misc import imsave
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle


#TODO Mask baselines before expanding the searchable region
#TODO Find a way to make a sorted list of coordinates from the image

#TODO 'Genetic' algorithm for finding templates (Not doing yet)


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

    def __init__(self, spectrogram, start_point, template, span=80, velo_scale=10):
        assert isinstance(spectrogram, Spectrogram)
        assert (len(template) > 0)

        self.spectrogram = spectrogram
        self.template = template[0]

        self.span = int(span)
        self.velo_scale = int(velo_scale)
        self.template_time_offset_index = int(template[1])
        self.template_velo_offset_index = int(template[2])


        self.zero_time_index = spectrogram._time_to_index(0)
        self.zero_velo_index = spectrogram._velocity_to_index(0)

        if start_point is None: 
            self.click = (self.zero_time_index, self.zero_velo_index)
        else:
            assert isinstance(start_point, tuple)
            self.click = start_point
        
        self.setup()



    def setup(self):

        velo_scale = self.velo_scale 
        time_index, velocity_index = self.click

        if time_index < 0:
            time_index = 0
        if velocity_index < 0:
            velocity_index = 0

        max_time_index = self.spectrogram.intensity.shape[1] - 1
        ending_time_index = time_index + self.span

        assert (ending_time_index < max_time_index)
        start_time = self.spectrogram.time[time_index]
        end_time = self.spectrogram.time[ending_time_index]

        max_velo_index = self.spectrogram.intensity.shape[0] - 1
        ending_velo_index = velocity_index + (velo_scale * self.span)

        assert (ending_velo_index < max_velo_index)
        start_velo = self.spectrogram.velocity[velocity_index]
        end_velo = self.spectrogram.velocity[ending_velo_index]

        zero_index = self.spectrogram._time_to_index(0)

        max_velo = self.spectrogram.velocity[max_velo_index]
        max_time = self.spectrogram.time[max_time_index]

        self.time_bounds = (time_index, ending_time_index) #indices, not actual time/velo values
        self.velo_bounds = (velocity_index, ending_velo_index)



    def crop_intensities(self, matrix):

        percentile_value = 90

        time_bounds = self.time_bounds
        velo_bounds = self.velo_bounds

        assert (velo_bounds[0] != velo_bounds[1])
        assert (time_bounds[0] != time_bounds[1])

        if velo_bounds[0] == 0: 
            flipped_velo_bounds = (-1*velo_bounds[1], -1)
        else:
            flipped_velo_bounds = (-1*velo_bounds[1], -1*velo_bounds[0])

        self.flipped_velo_bounds = flipped_velo_bounds

        sorted_matrix = sorted(matrix.flatten(), reverse=True)
        threshold_percentile = np.percentile(sorted_matrix, percentile_value)

        cleaned_matrix = np.where(matrix > threshold_percentile, matrix+threshold_percentile, threshold_percentile)
        # imsave("./original.png", cleaned_matrix[:])

        cleaned_matrix = np.flip(np.flip(cleaned_matrix), axis=1)
        # imsave("./flipped_original.png", cleaned_matrix[:])

        spec = cleaned_matrix[ flipped_velo_bounds[0]:flipped_velo_bounds[1], time_bounds[0]:time_bounds[1]]
        # imsave("./debug.png", spec[:])

        return spec



    def match(self):

        matrix = self.spectrogram.intensity
        cropped_spectrogram = self.crop_intensities(matrix)

        imsave("./im_template.png", self.template[:])
        imsave("./im_cropped_bg.png", cropped_spectrogram[:])

        img = cv2.imread('./im_cropped_bg.png', 0)
        img2 = img.copy()

        template = cv2.imread('./im_template.png', 0)
        w, h = template.shape[::-1]

        # All the 6 methods for comparison in a list
        methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
                    'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

        # methods = ['cv2.TM_CCORR_NORMED'] # the 'best' method for matching

        xcoords = []
        ycoords = []
        scores = []

        for meth in methods:
            img = img2.copy()
            method = eval(meth)

            # Apply template Matching
            res = cv2.matchTemplate(img, template, method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            
            # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                top_left = min_loc
                scores.append(min_val)
            else:
                top_left = max_loc
                scores.append(max_val)
            bottom_right = (top_left[0] + w, top_left[1] + h)

            velo_offset_index = self.template_velo_offset_index
            time_offset_index = self.template_time_offset_index

            real_velo_index = abs(self.flipped_velo_bounds[0] + bottom_right[1]) + velo_offset_index

            time_match = self.spectrogram.time[top_left[0]] * 1e6
            template_offset_time = self.spectrogram.time[time_offset_index] * 1e6
            start_time = self.spectrogram.time[self.zero_time_index] * 1e6 * -1
            time_offset = abs(self.spectrogram.time[self.time_bounds[0]] * 1e6)

            time_total = time_match + template_offset_time + start_time + time_offset

            true_velo = self.spectrogram.velocity[real_velo_index]

            xcoords.append(time_total)
            ycoords.append(true_velo)

            # cv2.rectangle(img, top_left, bottom_right, 255, thickness=1)
            # plt.subplot(121),plt.imshow(res,cmap = 'gray')
            # plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
            # plt.subplot(122),plt.imshow(img,cmap = 'gray')
            # plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
            # plt.suptitle(meth)
            # plt.show()

        return xcoords, ycoords, scores



if __name__ == "__main__":

    path = "/Users/trevorwalker/Desktop/Clinic/dig/new/WHITE_CH2_SHOT/seg00.dig"
    spec = Spectrogram(path, 0.0, 60.0e-6, overlap_shift_factor= 1/8, form='db')

    span = 200 # will determine the bounding box to search in

    template = opencv_long_start_pattern2 # use this template to search

    # gives user the option to click, by default it searches from (0,0)
    template_matcher = TemplateMatcher(spec, None, template, span=span)
    times, velos, scores = template_matcher.match()

    # draw the space to search in, plot times and velos as red dots
    dv = spec.velocity[template_matcher.velo_scale * span]
    dt = spec.time[span] * 1e6
    ax = plt.gca()
    spec.plot(ax)

    colors = ['ro', 'bo', 'go', 'mo', 'ko', 'co']
    color_names = ['red', 'blue', 'green', 'magenta', 'black', 'cyan']

    for i in range(len(times)):
        print("time: ", times[i])
        print("velocity: ", velos[i])
        print("score: ", scores[i])
        print("color: ", color_names[i],'\n')

        plt.plot(times[i], velos[i], colors[i], markersize=2.5, alpha=0.7)
    
    
    patch = Rectangle((0,0), dt, dv, fill=False, color='b', alpha=0.8)
    ax.add_patch(patch)

    # display image
    plt.show()


