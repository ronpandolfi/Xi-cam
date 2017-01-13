#! /usr/bin/env python

import glob
import os
import time

import numpy as np
import fabio
from scipy import optimize
from scipy import signal

import saxs_calibration
import peakfindingrem


def approx_width(r):
    """
    linearly varies the peak width between GISAXS, where peaks are
    thinner to GIWAXS, where peaks are fewer but wider.
    """
    return (0.047 * r + 1.8261)


def tophat2(radius, scale=1):
    """
    convolution kernel, revolved to form a ring, with Mexican Hat
    profile along the radial direction.

    radius : peak position along the radius
    scale  : magnification factor
    """
    width = approx_width(radius)
    N = np.round(radius) + 3 * round(width) + 1
    x = np.arange(-N, N)
    x, y = np.meshgrid(x, x)
    t = np.sqrt(x ** 2 + y ** 2) - radius
    s = width
    a = scale / np.sqrt(2 * np.pi) / s ** 3
    w = a * (1 - (t / s) ** 2) * np.exp(-t ** 2 / s ** 2 / 2.)
    return w


def calc_R(x, y, xc, yc):
    """ calculate the distance of each 2D points from the center (xc, yc) """
    return np.sqrt((x - xc) ** 2 + (y - yc) ** 2)


def f(c, x, y):
    """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
    Ri = calc_R(x, y, *c)
    return Ri - Ri.mean()


def fitpointstocircle(cnt):
    """Fit a Nx2 array of points to a circle; return center, radius, and fit residue"""

    # In case the shape is strange (i.e. extra dimensions) reshape it
    cnt = np.array(cnt.reshape((-1, 2)))

    # if the contour doesn't have enough points, give up
    if cnt.shape[0] < 3:
        return None, None, None, None

    # separate the points into x and y arrays
    x = cnt[:, 0]
    y = cnt[:, 1]

    # fit algorithm for circle from cookbook
    x_m = np.mean(x)
    y_m = np.mean(y)
    center_estimate = x_m, y_m
    center, ier = optimize.leastsq(f, center_estimate, args=(x, y))
    xc, yc = center
    Ri = calc_R(x, y, *center)
    R = Ri.mean()
    residu = np.sum((Ri - R) ** 2)
    return xc, yc, R, residu


# @debug.timeit
# def oldcenter_approx(img):
#     """
#     Find the center of scattering image; maskfit parameters are the Canny threshold parameters for sharp edges and
#     the Hough lines transform threshold; edgefitp parameters are the Canny thresholds for finding edges after masking
#     """
#     # newcenter_approx(img, experiment)
#     #return newcenter_approx(img, log=False)
#
#     try:
#         # Rescale brightness of the image with log depth
#         with np.errstate(divide='ignore', invalid='ignore'):
#             img = np.log(img) / np.log(np.max(img)) * 255
#
#         # Convert to 8bit image for the HoughCircles procedure
#         img = np.uint8(img)
#
#         # Histogram levels and equalize
#         img = cv2.equalizeHist(img)
#
#         # Draw for demo
#         if demo:
#             cv2.imshow('step?', cv2.resize(img, (0, 0), fx=.5, fy=.5))
#             cv2.waitKey(0)
#             cv2.destroyAllWindows()
#
#         # Add a 4 pixel gaussian blur to denoise the line search image
#         linesimg = cv2.GaussianBlur(img.copy(), (0, 0), 2)
#         linesimg = cv2.bilateralFilter(linesimg, -1, 10, 5)
#
#         # Do Canny edge search with mask parameters (looking for sharp edges to mask)
#         linesimg = cv2.Canny(linesimg, maskfitp[0], maskfitp[1])
#
#         kernel = np.ones((15, 15), np.uint8)
#         sharpmask = 1 - (cv2.dilate(cv2.morphologyEx(linesimg.copy(), cv2.MORPH_CLOSE, kernel), kernel) / 255)
#
#         # print sharpmask
#
#         # Draw for demo
#         if demo:
#             cv2.imshow('step?', cv2.resize(linesimg, (0, 0), fx=.5, fy=.5))
#             cv2.waitKey(0)
#             cv2.destroyAllWindows()
#
#             cv2.imshow('step?', cv2.resize(255 * sharpmask, (0, 0), fx=.5, fy=.5))
#             cv2.waitKey(0)
#             cv2.destroyAllWindows()
#
#         # Do Hough lines transform looking for lines at least as long as 1/4 the image width
#         #lines = cv2.HoughLines(linesimg, 1, np.pi / 180, 100, img.shape[0] / 4, maskfitp[2])
#
#         # Make an array for the mask
#         #linesmask = np.ones_like(img)
#
#         # for rho, theta in lines[0]:  # For each line found
#         #     a = np.cos(theta)
#         #     b = np.sin(theta)
#         #     x0 = a * rho
#         #     y0 = b * rho
#         #     x1 = int(x0 + 10000 * (-b))
#         #     y1 = int(y0 + 10000 * (a))
#         #     x2 = int(x0 - 10000 * (-b))
#         #     y2 = int(y0 - 10000 * (a))
#         #
#         #     # Mask out the line with a 20px bar
#         #     cv2.line(linesmask, (x1, y1), (x2, y2), (0, 0, 0), 20)
#         #
#         # # Draw for demo
#         # if demo == True:
#         #     cv2.imshow('step?', cv2.resize(linesmask * 255, (0, 0), fx=.5, fy=.5))
#         #     cv2.waitKey(0)
#         #     cv2.destroyAllWindows()
#
#         # Add a 3px gaussian blur to denoise the scattering image
#         img = cv2.GaussianBlur(img, (0, 0), 3)
#
#         # Add bilateral filtering to further denoise, preserving edges
#         img = cv2.bilateralFilter(img, -1, 10, 10)
#
#         # Cut the image at half its z-depth; this creates very good edges for Canny
#         img = img * 3
#
#         # Do Canny edge search to find rings; also apply the mask now
#         img = cv2.Canny(img, edgefitp[0], edgefitp[1]) * sharpmask
#
#         if demo:
#             cv2.imshow('detected circles', cv2.resize(img, (0, 0), fx=.5, fy=.5))
#             # wait to kill
#             cv2.waitKey(0)
#             cv2.destroyAllWindows()
#
#         # Make array to save circles
#         circles = []
#
#         # Find contours created by Canny; Each contour is a list of all chained points
#         contours, hierarchy = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#
#         for (h, cnt), cntindx in zip(enumerate(contours), range(
#                 contours.__len__())):  # For each contour; Zipped with a index to track each circle
#             # Fit the contour to a circle
#             x, y, R, residu = fitpointstocircle(cnt)
#
#             if x is not None and img.shape[0] > R > 200:  # If a circle was found and it has reasonable radius
#                 # Make a mask to select all additional points on the circle; If the circle was fit with only an arc,
#                 # this will include points further out
#                 mask = np.zeros_like(img)
#                 cv2.circle(mask, (int(x), int(y)), int(R), 255, 2)
#
#                 # Calculate how much of the full circle was found; this is the score for that circle
#                 coverage = (np.sum(img * mask) / (np.pi * 2 * R))
#
#                 # Add the circle to the list of circles found
#                 circles.append([x, y, R, coverage, cntindx])
#
#         # Make the circles array a nummpy array now that we're done building it
#         circles = np.array(circles)
#
#         # Sort and take the best circle (most coverage)
#         bestcircle = circles[circles[:, 3].argsort()[-1]]
#
#         # Draw the best guess for demo
#         if demo:
#             cv2.circle(img, (int(bestcircle[0]), int(bestcircle[1])), int(bestcircle[2]), 100, 5)
#
#         # Get all similar contours, looking for any circles that are within 20px radius and center
#         Rrange = 15
#         posrange = 15
#         similarcontours = []
#         for circle in circles:  # For each circle
#             # If the circle is close enough
#             if (bestcircle[2] - Rrange < circle[2] < bestcircle[2] + Rrange) \
#                     and (bestcircle[1] - posrange < circle[1] < bestcircle[1] + posrange) \
#                     and (bestcircle[0] - posrange < circle[0] < bestcircle[0] + posrange):
#                 # Add its points to a new binary image
#                 similarcontours.append(contours[int(circle[4])][:, 0, :])
#
#                 # Draw similar circles for demo
#                 if demo:
#                     cv2.circle(img, (int(circle[0]), int(circle[1])), int(circle[2]), 100, 5)
#
#         # Compress the array with vstack
#         similarcontours = np.vstack(similarcontours)
#
#         # Find the best circle fit from the new contour set
#         bestcircle = fitpointstocircle(similarcontours)
#
#         # Draw the finalized circle for demo
#         if demo:
#             cv2.circle(img, (int(bestcircle[0]), int(bestcircle[1])), int(bestcircle[2]), 255, 5)
#             cv2.imshow('detected circles', cv2.resize(img, (0, 0), fx=.5, fy=.5))
#             # wait to kill
#             cv2.waitKey(0)
#             cv2.destroyAllWindows()
#
#         # Return only the center
#         return bestcircle[0:2]
#     except Exception as ex:
#         template = "An exception of type {0} occured in center_approx. Arguments:\n{1!r}"
#         message = template.format(type(ex).__name__, ex.args)
#         print message
#         print "The center was unable to be found for an image."
#         return None

from xicam import debugtools


@debugtools.timeit
def center_approx(img, log=False):

    if log:
        # Rescale brightness of the image with log depth
        img = img.astype(np.float)
        with np.errstate(divide='ignore', invalid='ignore'):
            img = np.log(img * (img > 0) + 1)

    #testimg(img)

    con = signal.fftconvolve(img, img) / signal.fftconvolve(np.ones_like(img), np.ones_like(img))
    #testimg(con)

    cen = np.array(np.unravel_index(con.argmax(), con.shape)) / 2.
    #print('Center quality:',log,np.sum(con/con.max()))
    return cen


# def tth_ellipse(geometry, d_spacing):
# '''
# Returns a matplotlib.patches.Ellipse to plot the given d-spacing ring.
# '''
#     tth = 2 * np.arcsin(geometry.get_wavelength() / (2e-10 * d_spacing))
#
#     geo_dict = geometry.getFit2D()
#     sdd = geo_dict['directDist'] / (0.001 * geo_dict['pixelX'])
#     tilt = np.deg2rad(geo_dict['tilt'])
#     rotation = np.deg2rad(geo_dict['tiltPlanRotation'])
#
#     c_plus = (sdd * np.sin(tth)) / np.sin(np.pi / 2 - tilt - tth)
#     c_minus = (sdd * np.sin(tth)) / np.sin(np.pi / 2 + tilt - tth)
#     elli_h = (sdd * np.sin(tth)) / np.sin(np.pi / 2 - tth)
#     elli_w = (c_plus + c_minus) / 2.0
#
#     x_pos = (geo_dict['centerX'] - c_minus + elli_w) - geo_dict['centerX']
#     elli_x = geo_dict['centerX'] + x_pos * np.cos(rotation)
#     elli_y = geo_dict['centerY'] + x_pos * np.sin(rotation)
#
#     return matplotlib.patches.Ellipse((elli_x, elli_y),
#                                       elli_w * 2,
#                                       elli_h * 2,
#                                       np.rad2deg(rotation))


######################################################################################
######  REMI
###
#


def gisaxs_center_approx(img, log=False):
    img = img.astype(np.float)
    if log:
        # Rescale brightness of the image with log depth
        with np.errstate(divide='ignore', invalid='ignore'):
            img = np.log(img + 3) - np.log(3)

    x = 0
    xcenter = 0
    y = 10000
    ycenter = 0
    for i in range(0, img.shape[1]):
        if x <= sum(img[:, i]):
            x = sum(img[:, i])
            xcenter = i
        else:
            pass

    q = 4 * sum(img[img.shape[0] - 5, :])
    i = 0
    x = np.sum(img[:, :150], axis=1)
    for i in range(1, np.size(x)):
        if x[i] == 0:
            x[i] = x[i - 1]
        else:
            pass
    t = np.size(x) - 20
    x = signal.convolve(signal.convolve(x[:t], signal.gaussian(7, std=4)), [1, -1])
    # plt.plot(x)
    # plt.show()

    i = 0
    while (y != np.min(x)):
        y = x[i]
        ycenter = i - 6  # 6 because correct spread form convolution gaussian and derivation
        i = i + 1

    cen = (xcenter, ycenter)
    return cen



















    #
    ###
    ######  REMI


#########################################################################################################

def refinecenter(dimg):
    imgcopy = dimg.rawdata.T
    # Refine calibration
    # d-spacing for Silver Behenate
    d_spacings = np.array([58.367, 29.1835, 19.45567, 14.59175, 11.6734, 9.72783, 8.33814, 7.29587, 6.48522, 5.8367])

    geometry = dimg.experiment.getGeometry()

    # print 'Start parameter:'
    # print geometry.getFit2D()

    fit_param = ['center_x', 'center_y', 'distance', 'rotation', 'tilt']
    fit_thread = saxs_calibration.FitThread(geometry, d_spacings, imgcopy, fit_param, 40)
    fit_thread.start()
    while fit_thread.is_alive():
        #print fit_thread.status
        time.sleep(.1)

    # print 'Final parameter:'
    #print geometry.getFit2D()
    #print fit_thread.get_info()

    #pylab.imshow(np.log(img), interpolation='none')

    # for d_spacing in d_spacings:
    #    ellipse = tth_ellipse(geometry, d_spacing)
    #    ellipse.set_fc('none')
    #    ellipse.set_ec('red')
    #    pylab.gca().add_patch(ellipse)

    # for circle in fit_thread.get_circle_patches():
    #    pylab.gca().add_patch(circle)

    #pylab.show()

    return geometry.getFit2D()['centerX'], geometry.getFit2D()['centerY']

# def testimg(img, scale=0.5):
#     # Draw for demo
#     demo = False
#     if demo:
#         cv2.imshow('step?', cv2.resize((img * 255.0 / img.max()).astype(np.uint8), (0, 0), fx=.5, fy=.5))
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()


# if __name__ == "__main__":
# i = 0
# for imgpath in glob.glob(os.path.join("../../saxswaxs/AgBs", '*.edf')):
#         i += 1
#         if i < 0:
#             continue
#
#         print "Opening", imgpath
#
#         # read image
#         img = fabio.open(imgpath).data
#
#         outputimg = img.copy()
#
#         # print new center approximation; Add demo=True to see it work!
#         circle = center_approx(img, demo=True)
#         if circle is not None:
#             outputimg = np.uint8(outputimg)
#             outputimg = cv2.cvtColor(outputimg, cv2.COLOR_GRAY2BGR)
#             cv2.circle(outputimg, (int(circle[0]), int(circle[1])), int(circle[2]), (255, 0, 0), 3)
#             cv2.circle(outputimg, (int(circle[0]), int(circle[1])), 10, (255, 0, 0), 10)
#
#             cv2.imwrite(imgpath + "_center.png", cv2.resize(outputimg, (0, 0), fx=.3, fy=.3))
