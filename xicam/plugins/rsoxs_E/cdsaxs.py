# --coding: utf-8 --
import numpy as np
import matplotlib
from scipy.interpolate import LinearNDInterpolator
from astropy.modeling import models, fitting
from astropy.modeling.models import Const1D
from pyFAI import AzimuthalIntegrator,detectors
from scipy.ndimage import filters
from xicam import config
from scipy import signal
from pipeline import loader
import warnings
from astropy.utils.exceptions import AstropyUserWarning

def configu(a, file):
    if a == '733':
        data = loader.loadimage(file)
        img1 = np.rot90(loader.loadimage(file), 1)
        I_0 = loader.loadparas(data)['Izero']
        angle = loader.loadparas(data)['Sample Rotation Stage']
        #data = data / np.abs(I_0)
    elif a == '1102':
        data = np.flipud(loader.loadimage(file))            #Flipud works for RSOXS data
        img1 = np.rot90(loader.loadimage(file), 1)
        I_0 = loader.loadparas(file)['AI 3 Izero']
        angle = np.deg2rad(1.0 * (loader.loadparas(file)['sample theta'] - 90))
        wavelength = 10 ** -6 * 1.24 / loader.loadparas(file)['beamline energy']
        data = data / np.abs(I_0)
        data -= np.mean(data[ : -5, :])                     #Remove background
        data[np.where(data < 0)] = 0.000001
    else:
        data = loader.loadimage(file)
        img1 = np.rot90(loader.loadimage(file), 1)
        I_0 = 1
        #angle = phi
        #data = data / np.abs(I_0)
    return angle, wavelength, data, img1


def test(wavelength, substratethickness, substrateattenuation, Pitch, file):
    try:
        #data = np.flipud(loader.loadimage(file))
        beamline = '1102'
        phi, wavelength, data, img1 = configu(beamline, file)

        #636, 466
        np.save('/Users/guillaumefreychet/Desktop/data.npy', data)
        print(np.rad2deg(phi))

        q_n, I_cor = reduceto1dprofile(data, wavelength)
        #I_cor = correc_Iexp(I, substratethickness, substrateattenuation, phi)
        np.save('/Users/guillaumefreychet/Desktop/icor.npy', I_cor)

        I_peaks = Find_peak(q_n, I_cor, Pitch, wavelength)

        return I_cor, img1, I_peaks

    except IndexError:
        print 'Error in {}'.format(file)

def reduceto1dprofile(data, wavelength):
    beamline = '1102'
    if beamline == '1102':
        mask = config.activeExperiment.getDetector().calc_mask()
    elif beamline == '733':
        mask = (data > np.percentile(data, 98))
        mask = mask + config.activeExperiment.getDetector().calc_mask()
    else:
        #mask = (data > np.percentile(data, 98))
        mask = config.activeExperiment.getDetector().calc_mask()

    AI = AzimuthalIntegrator()
    AI = config.activeExperiment.getAI()
    #config.activeExperiment.setvalue('Wavelength', wavelength)

    np.save('/Users/guillaumefreychet/Desktop/data1.npy', data)

    cake, q, chi = AI.integrate2d(data, config.settings['Integration Bins (q)'], config.settings['Integration Bins (χ)'], mask=mask)
    cakemask, q, chi = AI.integrate2d(np.ones_like(data), config.settings['Integration Bins (q)'], config.settings['Integration Bins (χ)'])

    np.save('/Users/guillaumefreychet/Desktop/cake.npy', cake)

    maskedcake = np.ma.masked_array(cake, mask=cakemask <= 0)
    chiprofile = np.ma.average(maskedcake, axis=1)
    #np.save('/Users/guillaumefreychet/Desktop/chiprofile.npy', chiprofile.data)

    kernel = np.zeros_like(chiprofile)
    kernel[0] = 1
    kernel[len(kernel) / 2] = 1

    tiltprofile = filters.convolve1d(kernel, chiprofile, mode='wrap')
    np.save('/Users/guillaumefreychet/Desktop/tiltprofile.npy',tiltprofile)
    #tilt = tiltprofile[0 : int(0.2 * config.settings['Integration Bins (χ)'])].argmax() / float(config.settings['Integration Bins (χ)'])* 360.
    tilt = tiltprofile.argmax() / float(config.settings['Integration Bins (χ)'])* 360.

    slc = np.vstack([cake[:20,:],cake[-20:,:]]) # for other side, index around 500
    profile1=np.sum(slc,axis=0)


    q_n = np.array([q[val] for val in np.where(profile1 > 0)[0]])
    intensity = np.array([val for val in profile1 if val > 0])

    return q_n, intensity

def Find_peak(q, profiles, pitch, wavelength):

    np.save('/Users/guillaumefreychet/Desktop/q.npy', q)
    np.save('/Users/guillaumefreychet/Desktop/profile.npy', profiles)

    Qxexp, Q__Z, I_peaks = [[]] * 20, [[]] * 20, [[]] * 20
    q_pitch = np.abs(2. * np.pi / pitch)
    ind = np.argmin(np.abs(q - 2 * 1.5 * q_pitch))
    print(q_pitch, ind)

    #Fit gaussian
    pos_gauss = np.linspace(max(ind - 25, 0), min(ind + 25, len(profiles)-1), int(min(ind + 25, len(profiles)-1) - (max(ind - 25, 0))) - 1, dtype=np.int32)
    g_init = models.Gaussian1D(amplitude = profiles.max(), mean = ind, stddev = 3.)
    fit_g = fitting.LevMarLSQFitter()

    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        # Trigger a warning.
        g = fit_g(g_init, pos_gauss, profiles[pos_gauss])
        ind_ref = g.mean.value
        # Verify some things
        if len(w)>0:
            for x in w:
                if x.category == AstropyUserWarning:
                    ind_ref = profiles[pos_gauss].argmax() + max(ind - 25, 0)
        elif ind_ref<0:
            ind_ref = profiles[pos_gauss].argmax() + max(ind - 25, 0)
    print(g.mean.value, g.amplitude.value)

    q_peaks = []
    j = 0
    while abs(int((1 + 0.5 * j) * ind_ref)) < len(profiles) - 16:
        ind = int((1 + 0.5 * j) * ind_ref)
        #Switch to masked array
        pos_gauss1 = np.linspace(max(ind - 15, 0), min(ind + 15, len(profiles)), int(1 + min(ind + 15, len(profiles)) - (max(ind - 15, 0))), dtype=np.int32)
        g2 = models.Gaussian1D(amplitude=profiles[pos_gauss1].max(), mean=q[ind], stddev=0.001)
        g_init1 = g2
        fit_g = fitting.LevMarLSQFitter()
        gg = fit_g(g_init1, q[pos_gauss1], profiles[pos_gauss1])

        if gg.mean.value < min(q[pos_gauss1]) or gg.mean.value > max(q[pos_gauss1]):
            q_peaks.append(q[ind])
            I_peaks[j] = I_peaks[j] + [profiles[pos_gauss1].max()]
        else:
            q_peaks.append(gg.mean.value)
            I_peaks[j] = I_peaks[j] + [gg.amplitude.value]

        j += 1

    return I_peaks

# Correction of the footprint and substrate attenuation // Addition of sample size/sample attenuation and polarization
def correc_Iexp(pr, substratethickness, substrateattenuation, phi):
    footprintcorr = 'True'
    abscorr = 'True'
    samplesizecorr = 'False'
    fwhm, sample_size = 1, 1
    for i in range(0, len(pr), 1):
        footprintfactor = np.abs(np.cos(phi)) if footprintcorr else 1
        absfactor = np.exp(-substratethickness * substrateattenuation * (1 - 1 / np.cos(phi + 0.000000001))) if abscorr else 1
        pr[i] *= absfactor * footprintfactor
    return pr

def interpolation(q_x, q_z, I_i, sampling_size=(400, 400)):
    roi_size = 400
    img = np.zeros((roi_size, roi_size))

    qj = np.floor(((q_x - q_x.min()) / (q_x - q_x.min()).max()) * (sampling_size[0] - 1)).astype(np.int32)
    qk = np.floor(((q_z - q_z.min()) / (q_z - q_z.min()).max()) * (sampling_size[1] - 1)).astype(np.int32)
    I = I_i

    qj_shifted = qj - qj.min()
    qk_shifted = qk - qk.min()

    Isel = I
    for i in range(0, len(Isel), 1):
        img[qj_shifted[i], qk_shifted[i]] += Isel[i]

    log_possible = np.where(img!='nan')
    img[log_possible] = np.log(img[log_possible] - img[log_possible].min() + 1.)
    return img