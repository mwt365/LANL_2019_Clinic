

import cv2
import spectrogram as sp
import numpy as np
from imageio import imwrite
from matplotlib import pyplot as plt
from ImageProcessing.Templates.templates import *

# process command line args
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--file_name',type=str,default=False)
parser.add_argument('-m',default=False)
args = parser.parse_args()


#TODO read in original image once, give minMaxLoc for each template
#TODO Learn something from the yielded locations


def main():

    path = "/home/lanl/Documents/max/dig/sample.dig"
    spec = sp.Spectrogram(path, 0.0, 60.0e-6, overlap_shift_factor= 1/8, form='db')

    matrix = spec.intensity

    sortedmatrix = sorted(matrix.flatten(), reverse=True)

    threshold_percentile = np.percentile(sortedmatrix, 90)


    new_matrix = np.where(matrix > threshold_percentile, matrix+(2*threshold_percentile), 0)

    new_matrix = new_matrix[:1800]


    spec = np.flip(np.flip(new_matrix), axis=1)


    template1 = bigger_start_pattern


    imwrite("./template1.png", template1[:])
    imwrite("./photo.png", spec[:])


    # imread(path, 0) signifies reading in the image in grayscale mode
    img = cv2.imread('./photo.png',0)

    img2 = img.copy()
    template = cv2.imread('./template1.png',0)

    w, h = template.shape[::-1]

    # All the 6 methods for comparison in a lists
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
                'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    if args.m:

        methods = [methods[int(args.m)]]

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

        cv2.rectangle(img,top_left, bottom_right, 255, thickness=2)

        plt.subplot(121),plt.imshow(res,cmap = 'gray')
        plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(img,cmap = 'gray')
        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        plt.suptitle(meth)

        plt.show()


if __name__ == "__main__":

    main()