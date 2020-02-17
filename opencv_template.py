

import cv2
import spectrogram as sp
import numpy as np
from scipy.misc import imsave
from matplotlib import pyplot as plt
from ImageProcessing.Templates.templates import *



# TODO Learn something from the yielded locations
# TODO find threshold value and set every pixel to either 1 or 0
# TODO use K-means to classify clusters into 5 colors before searching for templates.


#TODO Have the user click
#TODO Slowly increase the box radius 
#TODO Plot the white box inside the red box
#TODO get trace extraction working with this clicking

#TODO Little successes with human interaction before fully autonomous system. 

#TODO Supervised learning with clicks to find templates



def main():

    path = "/Users/trevorwalker/Desktop/Clinic/For_Candace/newdigs/CH_2_009.dig"
    spec = sp.Spectrogram(path, 0.0, 60.0e-6, overlap_shift_factor= 1/8, form='db')

    matrix = spec.intensity

    print(matrix.shape, '\n')

    sortedmatrix = sorted(matrix.flatten(), reverse=True)
    threshold_percentile = np.percentile(sortedmatrix, 90)

    new_matrix = np.where(matrix > threshold_percentile, matrix+threshold_percentile, threshold_percentile)
    new_matrix = new_matrix[400:1000, 50:300]

    print(new_matrix.shape)

    spec = np.flip(np.flip(new_matrix), axis=1)

    template1 = opencv_start_pattern

    imsave("./template1.png", template1[:])
    imsave("./photo.png", spec[:])

    # imread(path, 0) signifies reading in the image in grayscale mode
    img = cv2.imread('./photo.png', 0)

    img2 = img.copy()
    template = cv2.imread('./template1.png', 0)

    w, h = template.shape[::-1]

    # All the 6 methods for comparison in a list
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
                'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

    # methods = ['cv2.TM_CCOEFF']

    res_methods = []

    for meth in methods:
        img = img2.copy()
        method = eval(meth)

        # Apply template Matching
        res = cv2.matchTemplate(img,template,method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        print(max_loc, max_val)

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
    # template1 = opencv_start_pattern

    # imsave("./template1.png", template1[:])