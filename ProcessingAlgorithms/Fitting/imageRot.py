from scipy import ndimage, misc
import matplotlib.pyplot as plt
import numpy as np


def computeFit(xArray, yArray):
    """
        Input:
            xArray- List of the past2 x coordinates
            yArray- List of the past2 y coordinates
            (time value, velocity value)
    """
    myLine = np.polynomial.polynomial.Polynomial.fit(xArray, yArray, 1)
    offset, m  = myLine.convert().coef

    return [offset, m]


    
def computeAngle(coefficents):
    """
        Input:
            coefficients: The slope and intercept of the best fit line.
    """
    offset, m  = coefficents

    return -1*np.degrees(np.arctan2(1, m))

def rotate(intensityMatrix, angleToRotate, **kwargs):
    return ndimage.rotate(intensityMatrix, angleToRotate, reshape=False, **kwargs)

def extrapolate(coefficents, xValue):
    """
        Input:
            coefficients: The slope and intercept of the best fit line.
            xValue: The value to compute the y value for.
    """
    offset, m = coefficents
    return offset + m*xValue


def computeIndices(org_center:tuple, rot_center:tuple, xyPairs, angle, org_box_offset:tuple):
    """
        Angle is assumed to be in degrees.
    """
    rot_center = np.array([[rot_center[0]], [rot_center[1]]])
    org_center = np.array([[org_center[0]], [org_center[1]]])
    org_box_offset = np.array([[org_box_offset[0]], [org_box_offset[1]]])
    newPairs = xyPairs - rot_center

    rotMatrix = buildRotationMatrix(angle)

    orgPairs = np.matmul(rotMatrix, newPairs)

    return np.array(orgPairs) + org_center + org_box_offset

def calculateNewVelocityAndTime(dV:float, dt:float, t_start:float, v_start:float, velTimeIndices:np.array):

    velocities = velTimeIndices[0]*dV + v_start
    time = velTimeIndices[1]*dt + t_start

    return velocities, time

def buildRotationMatrix(angle):
    a = np.radians(angle)

    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])



if __name__ == "__main__":
    fig = plt.figure(figsize=(10, 3))
    ax1, ax2, ax3 = fig.subplots(1, 3)
    img = misc.ascent()

    print(img.shape)


    img_30 = ndimage.rotate(img, 30, reshape=False)

    xyPairs = np.array([(x,y) for x in range(img_30.shape[0]) for y in range(img_30.shape[1])])
    xyPairs = xyPairs.transpose()
    print(xyPairs.shape)

    q = computeIndices((np.array(img.shape[::-1])-1)/2, (np.array(img_30.shape[::-1])-1)/2, xyPairs, -30, (0,0))

    print(q)

    full_img_60 = ndimage.rotate(img, 60, reshape=False)
    ax1.imshow(img, cmap='gray')
    ax1.set_axis_off()
    ax2.imshow(img_30, cmap='gray')
    ax2.set_axis_off()
    ax3.imshow(full_img_60, cmap='gray')
    ax3.set_axis_off()
    fig.set_tight_layout(True)
    plt.show()