import glob
import os

import numpy as np
import scipy
from scipy import signal
import fabio
import matplotlib.pyplot as plt

import center_approx
import integration

import peakfindingrem
import peakfinding
import scipy.optimize as optimize


demo = True


# def find_arcs(img, cen):
# img = img * (img > 1)
#     img = np.log(img + 1) / np.log(np.max(img)) * 255
#     img = np.array(img, dtype=np.uint8)
#     img = img * Pilatus2M_Mask()
#
#     plt.imshow(img)
#     plt.show()
#     # plt.imshow(img)
#     # plt.show()
#     # search theta
#     Nradial = 25
#
#     arclist = []
#
#     for Theta in np.linspace(-180.0, 0.0, Nradial, endpoint=False):
#         mask = arcmask(img, cen, [0, img.shape[0]], [Theta, Theta + 180.0 / Nradial])
#
#         if demo:
#             plt.imshow(img * mask)
#             plt.show()
#
#         thetaprofile = radial_integrate(mask * img, cen)
#         # print thetaprofile
#
#         peakRs = findpeaks(thetaprofile)
#
#         print peakRs
#
#         if demo:
#             fig = plt.figure()
#
#             h = fig.add_subplot(211)
#             plt.plot(thetaprofile)
#             plt.plot(peakRs, thetaprofile[peakRs], 'r*')
#             limits = h.axis()
#             h = fig.add_subplot(212)
#             plt.imshow(signal.cwt(thetaprofile, signal.ricker, np.arange(5, 100)))
#             cwtlimits = h.axis()
#             h.axis([limits[0], limits[1], cwtlimits[2], cwtlimits[3], ])
#             fig.tight_layout()
#             plt.show()
#
#         for peakR in peakRs:
#             Rwidth = 20
#             mask = arcmask(img, cen, (-0.5 * Rwidth + peakR, 0.5 * Rwidth + peakR), (-180.0, 0.0))
#             if demo:
#                 plt.imshow(img * mask)
#                 plt.show()
#
#             Rprofile = alt_integrate(img * mask, cen)
#
#             Nangle = 50.0
#             peakThetas = np.array(findpeaks(Rprofile))
#
#             if demo:
#                 plt.plot(np.linspace(0, np.pi * 2, Rprofile.__len__()), Rprofile)
#                 plt.plot(peakThetas / Nangle, Rprofile[peakThetas], 'r*')
#                 plt.show()
#
#             peakThetas = peakThetas / Nangle
#             print 'Here', peakThetas
#
#             #add peak to arclist
#             for peakTheta in peakThetas:
#                 arclist.append([peakR * np.cos(peakTheta), peakR * np.sin(peakTheta)])
#
#     arclist = np.array(arclist)
#     plt.imshow(img)
#     plt.plot(cen[0] - arclist[:, 0], cen[1] - arclist[:, 1], 'r*')
#     plt.show()


def findpeaks(Y):
    # Find peaks using continuous wavelet transform; parameter is the range of peak widths (may need tuning)
    # plt.plot(Y)
    # plt.show()
    Y = np.nan_to_num(Y)
    peakindices = scipy.signal.find_peaks_cwt(Y, np.arange(20, 100), noise_perc=5, min_length=10)

    # peakindices=peakutils.indexes(Y, thres=0.6,min_dist=20)

    return peakindices


# def radial_integrate(img, cen):
# # Radial integration
#     y, x = np.indices(img.shape)
#     r = np.sqrt((x - cen[0]) ** 2 + (y - cen[1]) ** 2)
#     r = np.rint(r).astype(np.int)
#
#     tbin = np.bincount(r.ravel(), img.ravel())
#
#     nr = np.bincount(r.ravel(), (img > 0).ravel())
#     radialprofile = tbin / nr
#
#     return radialprofile


# def alt_integrate(img, cen):
# # Radial integration
#     N = 50
#     y, x = np.indices((img.shape))
#     r = (np.arctan2(y - cen[1], x - cen[0]) + np.pi) * N
#     r = np.rint(r).astype(np.int)
#     tbin = np.bincount(r.ravel(), img.ravel())
#     nr = np.bincount(r.ravel(), (img > 0).ravel())
#     radialprofile = tbin / nr
#
#     return radialprofile


# def Pilatus2M_Mask():
# row = 1679
#     col = 1475
#     mask = np.zeros((row, col))
#
#     row_start = 196
#     row_gap_size = 16
#     row_num_gaps = 7
#
#     col_start = 488
#     col_gap_size = 6
#     col_num_gaps = 2
#
#     start = row_start
#     for i in range(1, row_num_gaps + 1):
#         mask[start:start + row_gap_size, :] = 1
#         start = start + row_gap_size + row_start
#     start = col_start
#     for j in range(1, col_num_gaps + 1):
#         mask[:, start:start + col_gap_size] = 1
#         start = start + col_gap_size + col_start
#
#     return 1 - mask


# def arcmask(img, cen, Rrange, Thetarange):
# mask = np.zeros_like(img)
# #print cen, Rrange,Thetarange
# if min(Rrange)==0:
# cv2.ellipse(mask, (int(cen[0]),int(cen[1])), (int(max(Rrange)),int(max(Rrange))), 0, min(Thetarange), int(max(Thetarange)), 255, -1)
#     else:
#         cv2.ellipse(mask, (int(cen[0]),int(cen[1])), (int(min(Rrange)),int(min(Rrange))), 0, min(Thetarange), int(max(Thetarange)), 255, int(max(Rrange)-min(Rrange)))
#     #cv2.ellipse(mask,(256,256),(100,50),0,0,180,255,-1)
#
#     plt.imshow(mask)
#     plt.show()
#     return mask/255

def arcmask(img, cen, Rrange, Thetarange):
    mask = np.zeros_like(img)

    y, x = np.indices((img.shape))
    r = np.sqrt((x - cen[0]) ** 2 + (y - cen[1]) ** 2)
    theta = np.arctan2(y - cen[1], x - cen[0]) / (2 * np.pi) * 360.0
    mask = ((min(Rrange) < r) & (r < max(Rrange)) & (min(Thetarange) < theta) & (theta < max(Thetarange)))
    #plt.imshow(mask)
    #plt.show()

    return mask


def scanforarcs(radialprofile, cen):
    # h = 35
    # radialprofile=signal.convolve(radialprofile,signal.gaussian(h, std=8))
    # test = np.max(radialprofile) / h
    #print 't', test
    peakmax, peakmin = peakfindingrem.peakdet(range(len(radialprofile)), radialprofile, 10)
    peakind = peakmax[:, 0]


    # for i in range(np.size(peakind)):
    # plt.axvline(peakind[i],color='b')
    # plt.plot(radialprofile)
    # plt.show()

    # accurancy = 50
    # x = np.zeros((np.size(peakind), accurancy))
    #y = np.zeros((np.size(peakind), accurancy))
    #xinf = np.zeros((np.size(peakind), accurancy))
    #yinf = np.zeros((np.size(peakind), accurancy))
    #xsup = np.zeros((np.size(peakind), accurancy))
    #ysup = np.zeros((np.size(peakind), accurancy))

    # for i in range(0, np.size(peakind)):
    #Delta = peakind[i] / 10
    #theta = np.linspace(0, 2 * np.pi, accurancy)
    #x[i] = cen[0] + (peakind[i]) * np.cos(theta)
    #y[i] = cen[1] + (peakind[i]) * np.sin(theta)
    #xinf[i] = cen[0] + (peakind[i] - Delta) * np.cos(theta)
    #xsup[i] = cen[0] + (peakind[i] + Delta) * np.cos(theta)
    #yinf[i] = cen[1] + (peakind[i] - Delta) * np.sin(theta)
    #ysup[i] = cen[1] + (peakind[i] + Delta) * np.sin(theta)

    return peakind


def gaussian(x, a, b, c, d):
    val = abs(a) * np.exp(-(x - b) ** 2 / c ** 2) + abs(d)
    return val


tworoot2ln2 = 2 * np.sqrt(2 * np.log(2))


def findgisaxsarcs(img, cen, experiment):
    radialprofile = integration.pixel_2Dintegrate(img, (cen[1], cen[0]), experiment.mask)
    # arcs = scanforarcs(radialprofile, cen)
    arcs = peakfinding.findpeaks(None, radialprofile, (100, 50), gaussianwidthsigma=3, minimumsigma=100)
    # print arcs
    plt.plot(radialprofile)
    plt.plot(arcs[0], arcs[1], 'ok')
    plt.show()
    arcs = arcs[0]

    output = []
    for qmu in arcs:
        chiprofile = integration.chi_2Dintegrate(img, (cen[1], cen[0]), qmu, mask=experiment.mask)
        try:
            popt, pcov = optimize.curve_fit(gaussian, np.arange(np.size(chiprofile)), np.nan_to_num(chiprofile))
        except RuntimeError:
            continue
        perr = np.sqrt(np.abs(np.diag(pcov)))
        # print 'perr:',perr
        if np.any(perr > 10):
            pass
            # print 'Parameter error too large, discarding arc at qmu'
            #continue
        chimu, A, sigma, baseline = popt
        FWHM = sigma * tworoot2ln2
        output.append([qmu, chimu, A, FWHM, baseline])
        # plt.plot(chiprofile)
        # plt.plot(gaussian(np.arange(np.size(chiprofile)), *popt))
        # plt.show()

    return output





if __name__ == "__main__":
    import hipies.config

    experiment = hipies.config.experiment()
    experiment.setvalue('Detector', 'pilatus2m')
    experiment.mask = experiment.getDetector().calc_mask()

    for imgpath in glob.glob(os.path.join("../GISAXS samples/", '*.edf')):
        print "Opening", imgpath

        # read image
        img = fabio.open(imgpath).data
        # find center
        # cen = center_approx.center_approx(img)

        cen = center_approx.gisaxs_center_approx(img)
        arcs = findgisaxsarcs(img, cen, experiment)
        # print cen
        #print arcs

        plt.axvline(cen[0], color='r')
        plt.axhline(cen[1], color='r')
        plt.imshow(np.log(img))

        ax = plt.axes()

        from matplotlib.patches import Arc

        for arc in arcs:
            print arc[3]
            arcs = [Arc(xy=cen, width=arc[0] * 2, height=arc[0] * 2, angle=-90, theta1=0,
                        theta2=abs(arc[3]) / (400 * np.pi / 2) * 360 * 4)]  # Arc
            ax.add_artist(arcs[0])
            arcs[0].set_lw(3)


        # for i in range(1, np.size(x, 0)):
        # plt.plot(y[i], x[i], color='g')
        #     plt.plot(yinf[i], xinf[i], color='r')
        #     plt.plot(ysup[i], xsup[i], color='r')
        plt.show()

        # popt, pcov = optimize.curve_fit(gaussian, np.arange(np.size(a)), np.nan_to_num(a))

        # print("Scale =  %.3f +/- %.3f" % (popt[0], np.sqrt(pcov[0, 0])))
        #print("Offset = %.3f +/- %.3f" % (popt[1], np.sqrt(pcov[1, 1])))
        #print("Sigma =  %.3f +/- %.3f" % (popt[2], np.sqrt(pcov[2, 2])))

        # print(vimodel.A)
        # print vimodel.mu
        # print vimodel.FWHM




        # find arcs
        # arcs = find_arcs(img, cen)

        #draw arcs
        #drawarcs(img,arcs)
