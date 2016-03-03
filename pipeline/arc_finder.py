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
import scipy.stats
from scipy import interpolate

from lmfit import minimize, Parameters
import cv2

from scipy.ndimage import filters

from scipy.fftpack import rfft, irfft

from skimage.restoration import denoise_bilateral


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


def mirroredgaussian(theta, a, b, c, d):
    val = (gaussian(theta, a, b, c, d) + gaussian(2 * np.pi - theta, a, b, c, d)) / 2.
    return val

def gaussian(x, a, b, c, d):
    val = abs(a) * np.exp(-(x - b) ** 2. / c ** 2.) + abs(d)
    return val


def vonmises(x, A, mu, kappa):
    return A * scipy.stats.vonmises.pdf(2 * (x - mu), kappa)


def mirroredvonmises(x, A, mu, kappa, floor):
    return A * (scipy.stats.vonmises.pdf(2 * (mu - x), kappa) + scipy.stats.vonmises.pdf(2 * (mu - x),
                                                                                         kappa)) / 2 + floor  # 2*(mu-(np.pi-x))


tworoot2ln2 = 2. * np.sqrt(2. * np.log(2.))


def residual(params, x, data):
    A = params['A'].value
    mu = params['mu'].value
    kappa = params['kappa'].value
    floor = params['floor'].value

    model = mirroredvonmises(x, A, mu, kappa, floor)

    return data - model


def gaussianresidual(params, x, data, sig=1):
    A = params['A'].value
    mu = params['mu'].value
    sigma = params['sigma'].value
    floor = params['floor'].value

    model = gaussian(x, A, mu, sigma, floor)

    resids = data - model

    # print resids

    weighted = np.sqrt(resids ** 2 / sig ** 2)
    return weighted


def findgisaxsarcs(img, cen, experiment):
    radialprofile = integration.pixel_2Dintegrate(img, (cen[1], cen[0]), experiment.mask)
    # arcs = scanforarcs(radialprofile, cen)
    arcs = peakfinding.findpeaks(None, radialprofile, (100, 50), gaussianwidthsigma=3, minimumsigma=100)
    # print arcs
    plt.plot(radialprofile)
    plt.plot(arcs[0], arcs[1], 'ok')
    #plt.show()
    arcs = arcs[0]

    output = []
    _, unique = np.unique(arcs, return_index=True)

    for qmu in arcs[unique]:
        chiprofile = np.nan_to_num(integration.chi_2Dintegrate(img, (cen[1], cen[0]), qmu, mask=experiment.mask))

        plt.plot(np.arange(0, np.pi, 1 / 30.), chiprofile, 'r')

        # filter out missing chi
        missingpointfloor = np.percentile(chiprofile, 15)
        badpoints = np.where(chiprofile < missingpointfloor)[0]
        goodpoints = np.where(chiprofile >= missingpointfloor)[0]

        chiprofile[badpoints] = np.interp(badpoints, goodpoints, chiprofile[goodpoints])

        plt.plot(np.arange(0, np.pi, 1 / 30.), chiprofile, 'k')

        # f=rfft(chiprofile)
        # plt.plot(f)
        # f[-20:]=0
        # chiprofile=irfft(chiprofile)




        try:
            params = Parameters()
            params.add('A', value=np.max(chiprofile), min=0)
            params.add('mu', value=np.pi / 2, min=0, max=np.pi)
            params.add('kappa', value=0.1, min=0)
            params.add('floor', value=0.1, min=0)
            x = np.arange(0, np.pi, 1 / 30.)

            out = minimize(residual, params, args=(x, chiprofile))
            print params
            # print params['A'].stderr

            # popt, pcov = optimize.curve_fit(vonmises, np.arange(0, np.pi, 1 / 30.), np.nan_to_num(chiprofile),
            #

            # print(popt)
        except RuntimeError:
            print('Fit failed at ' + qmu)
            continue

        if params['kappa'].stderr > 100 or params['A'].stderr > 100:
            isring = True
        else:
            isring = False

        popt = [params['A'].value, params['mu'].value, params['kappa'].value, params['floor'].value]
        A, chimu, kappa, baseline = popt
        FWHM = np.arccos(np.log(.5 * np.exp(kappa) + .5 * np.exp(-kappa)) / kappa)

        output.append([qmu, A, chimu, FWHM, baseline, isring])
        # plt.plot(np.arange(0, np.pi, 1 / 30.), chiprofile)
        # plt.plot(np.arange(0, np.pi, 1 / 30.), mirroredvonmises(np.arange(0, np.pi, 1 / 30.), *popt))
        # plt.show()

    return output


def inpaint(img,mask):
    filled = None
    if False:
        img = img / (2 ^ 16 - 1) * 255
        plt.imshow(img.astype(np.uint8))
        plt.show()

        plt.imshow(mask)  #TODO: check that mask corners are correct
        plt.show()

        kernel = np.ones((3, 3),np.uint8)
        mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)

        filled = cv2.inpaint(img.astype(np.uint8), mask.astype(np.uint8), 3, cv2.INPAINT_TELEA)

        plt.imshow(img)
        plt.show()

        return

    elif True:

        valid = ~mask.astype(np.bool)
        coords = np.array(np.nonzero(valid)).T
        values = img[valid]

        it = interpolate.LinearNDInterpolator(coords, values)

        filled = it(list(np.ndindex(img.shape))).reshape(img.shape)

        plt.imshow(np.rot90(filled))
        plt.show()

    return filled


def findmaxs(orig):
    img = orig.copy()
    img = filters.gaussian_filter(img, 3)
    img = filters.minimum_filter(img, 4)

    img = filters.median_filter(img, 4)

    img -= np.min(img)
    img = denoise_bilateral(img, sigma_range=0.5, sigma_spatial=15)

    # plt.imshow(np.rot90(img))
    #plt.show()

    #img = filters.percentile_filter(img,50,50)

    maxima = ((img == filters.maximum_filter(img, (10, 10))) & (
    filters.maximum_filter(img, (50, 50)) > 1.5 * filters.minimum_filter(img, (50, 50))) & (img > 2))
    maximachis, maximaqs = np.where(maxima == 1)
    plt.imshow(np.rot90(orig), interpolation='nearest')
    plt.plot(maximachis, 1000 - maximaqs, 'o', markersize=10, markeredgecolor='red', markerfacecolor="None", mew="4")
    plt.ylim([1000, 0])
    plt.xlim([0, 1000])
    plt.show()

    return maximachis, maximaqs


def fitarc(chiprofile):
    try:
        params = Parameters()
        params.add('A', value=np.max(chiprofile), min=0)
        params.add('mu', value=np.pi / 2, min=0, max=np.pi)
        params.add('kappa', value=0.1, min=0)
        params.add('floor', value=0.1, min=0)
        x = np.arange(0, np.pi, 1 / 30.)

        out = minimize(residual, params, args=(x, chiprofile))
        print params
        # print params['A'].stderr

        # popt, pcov = optimize.curve_fit(vonmises, np.arange(0, np.pi, 1 / 30.), np.nan_to_num(chiprofile),
        # p0=[np.max(np.nan_to_num(chiprofile)), np.pi / 2, .1, 0])
        # print(popt)
    except RuntimeError:
        print('Fit failed.')

    if params['kappa'].stderr > 100 or params['A'].stderr > 100:
        isring = True
    else:
        isring = False

    popt = [params['A'].value, params['mu'].value, params['kappa'].value, params['floor'].value]
    A, chimu, kappa, baseline = popt
    FWHM = np.arccos(np.log(.5 * np.exp(kappa) + .5 * np.exp(-kappa)) / kappa)

    return A, chimu, FWHM, baseline, isring


def fitarcgaussian(chiprofile, chi):
    try:
        params = Parameters()
        x = np.arange(np.size(chiprofile))
        roi = np.ones_like(chiprofile)
        roi[chi - 30:chi + 30] = .0001
        # roi/=1000
        # plt.plot(chiprofile,'')
        #plt.plot(roi * np.max(chiprofile * roi), 'g')
        #plt.plot(roi*chiprofile,'k')
        params.add('A', value=np.max(chiprofile * (1 - roi)), min=0)
        params.add('mu', value=chi, min=0, max=len(chiprofile))
        params.add('sigma', value=20, min=0)
        params.add('floor', value=0.1, min=0)

        out = minimize(gaussianresidual, params, args=(x, chiprofile, roi), method='nelder')
        #print params
        # print params['A'].stderr

        # popt, pcov = optimize.curve_fit(vonmises, np.arange(0, np.pi, 1 / 30.), np.nan_to_num(chiprofile),
        # p0=[np.max(np.nan_to_num(chiprofile)), np.pi / 2, .1, 0])
        # print(popt)
    except RuntimeError:
        print('Fit failed.')

    if params['sigma'].stderr > 100 or params['A'].stderr > 100:
        isring = False  #True
    else:
        isring = False

    popt = [params['A'].value, params['mu'].value, params['sigma'].value, params['floor'].value]
    #plt.plot(x, gaussian(x, *popt), 'r')
    # plt.show()
    # A, chimu, sigma, baseline = popt
    # FWHM = sigma * tworoot2ln2

    return popt


def findgisaxsarcs2(img, experiment):
    img = img.T.copy()
    cake, _, _ = integration.cake(img, experiment, mask=experiment.mask)  # TODO: refactor these parameters and check .T
    maskcake, _, _ = integration.cake(experiment.mask.T, experiment)

    from fabio import edfimage

    fabimg = edfimage.edfimage(cake)
    filename = 'cake.edf'
    fabimg.write(filename)

    fabimg = edfimage.edfimage(maskcake)
    filename = 'cake_MASK.edf'
    fabimg.write(filename)

    img = inpaint(cake, maskcake)

    fabimg = edfimage.edfimage(img)
    filename = 'cake_LINEAR_INFILL.edf'
    fabimg.write(filename)

    maxchis, maxqs = findmaxs(img)

    out =[]

    for chi, q in zip(maxchis, maxqs):
        # roi=np.ones_like(img)
        #roi[chi - 10:chi + 10, q - 5:q + 5]=10
        #roi=np.sum(roi,axis=1)
        slice = img[:, q - 5:q + 5]
        if np.max(slice) / np.min(slice) < 2:
            pass  # continue
        chiprofile = np.sum(slice, axis=1)
        x = np.arange(np.size(chiprofile))

        #plt.plot(chiprofile)

        params = fitarcgaussian(chiprofile, chi)
        if params['mu'] > chi + 5 or params['mu'] < chi - 5:
            continue
        params.add('q', value=q)
        out.append(params)



        #plt.show()

    # plt.imshow(np.log(img))
    #plt.show()

    return out


if __name__ == "__main__":
    import xicam.config

    experiment = xicam.config.experiment()
    experiment.setvalue('Detector', 'pilatus2m')
    experiment.setvalue('Pixel Size X',172e-6)
    experiment.setvalue('Pixel Size Y', 172e-6)
    experiment.mask = experiment.getDetector().calc_mask()

    for imgpath in glob.glob(os.path.join("../GISAXS samples/", '*.edf')):
        print "Opening", imgpath

        # read image
        img = fabio.open(imgpath).data
        # find center
        # cen = center_approx.center_approx(img)



        cen = center_approx.gisaxs_center_approx(img)
        experiment.setcenter(cen)

        arcs = findgisaxsarcs2(img, experiment)
        # print cen
        # print arcs

        ax = plt.gca()

        plt.axvline(cen[0], color='r')
        plt.axhline(cen[1], color='r')
        plt.imshow(np.log(img))



        from matplotlib.patches import Arc

        qratio =1.78

        for arc in arcs:
            print arc
            if not np.isnan(arc['sigma'].value):

                if False:
                    arcartist = [Arc(xy=cen, width=arc['q'] * 2, height=arc['q'] * 2, angle=-90, theta1=0,
                                     theta2=360)]  # Arc
                    ax.add_artist(arcartist[0])
                    arcartist[0].set_lw(3)
                else:
                    angle = -arc['mu'].value / 1000 * 360
                    theta1 = -abs(arc['sigma'].value * tworoot2ln2) / 1000 * 360 / 2
                    theta2 = abs(arc['sigma'].value * tworoot2ln2) / 1000 * 360 / 2
                    arcartist = [
                        Arc(xy=cen, width=arc['q'].value * 2 * qratio, height=arc['q'].value * 2 * qratio, angle=angle,
                            theta1=theta1,
                            theta2=theta2)]  # Arc
                    for artist in arcartist:
                        ax.add_artist(artist)
                        artist.set_lw(3)


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
