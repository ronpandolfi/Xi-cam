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

def test(wavelength, substratethickness, substrateattenuation, Pitch, file):
    try:
        #file, phi = Data[0], Data[1]
        #data = np.flipud(loader.loadimage(file))


        data = loader.loadimage(file)
        img1 = np.rot90(loader.loadimage(file),3)

        I_0 = loader.loadparas(file)['AI 3 Izero']
        energy = loader.loadparas(file)['Beamline Energy']

        wavelength = 4.14 * 10**-15 * 299792458 / energy
        #636, 466

        data = data / (np.abs(I_0 * 1000.))
        #DARK correction
        data -= (np.mean(data[0:5, :]) - 2)
        np.save('/Users/guillaumefreychet/Desktop/data.npy', data)

        q_n, I = reduceto1dprofile(data, wavelength)
        np.save('/Users/guillaumefreychet/Desktop/icor.npy', I)

        q_peaks, I_peaks, wavel = Find_peak(q_n, I, Pitch, wavelength)

        return I, img1, q_peaks, I_peaks, wavel

    except IndexError:
        print 'Error in {}'.format(file)

def reduceto1dprofile(data, wavelength):
    beamline = '1102'
    if beamline == '1102':
        mask = config.activeExperiment.getDetector().calc_mask()

    #config.activeExperiment.getvalue('Wavelength')

    AI = AzimuthalIntegrator()
    AI = config.activeExperiment.getAI()

    AI.wavelength = wavelength

    method = 'None'
    cake, q, chi = AI.integrate2d(data, config.settings['Integration Bins (q)'], config.settings['Integration Bins (χ)'], mask=mask, method=method)
    cakemask, q, chi = AI.integrate2d(np.ones_like(data), config.settings['Integration Bins (q)'], config.settings['Integration Bins (χ)'], mask=mask, method=method)

    np.save('/Users/guillaumefreychet/Desktop/cake.npy', cake)

    maskedcake = np.ma.masked_array(cake, mask=cakemask <= 0)
    chiprofile = np.ma.average(maskedcake, axis=1)
    #np.save('/Users/guillaumefreychet/Desktop/chiprofile.npy', chiprofile.data)

    kernel = np.zeros_like(chiprofile)
    kernel[0] = 1
    kernel[len(kernel) / 2] = 1

    tiltprofile = filters.convolve1d(kernel, chiprofile, mode='wrap')
    np.save('/Users/guillaumefreychet/Desktop/tiltprofile.npy',tiltprofile)
    tilt = tiltprofile[0 : int(0.2 * config.settings['Integration Bins (χ)'])].argmax() / float(config.settings['Integration Bins (χ)'])* 360.
    #print('tilt', tilt)

    # correct tilt
    AI.set_rot3(np.deg2rad(tilt))
    cake, q, chi = AI.integrate2d(data, config.settings['Integration Bins (q)'], config.settings['Integration Bins (χ)'], mask=mask, method=method)
    cakemask, q, chi = AI.integrate2d(np.ones_like(data), config.settings['Integration Bins (q)'], config.settings['Integration Bins (χ)'], mask=mask, method=method)
    np.save('/Users/guillaumefreychet/Desktop/cake1.npy', cake)

    slc = np.vstack([cake[:5,:],cake[-5:,:]]) # for other side, index around 500
    profile1=np.sum(slc,axis=0)

    q_n = np.array([q[val] for val in np.where(profile1 > 0)[0]])
    intensity = np.array([val for val in profile1 if val > 0])

    return q_n, intensity

def Find_peak(q, profiles, pitch, wavelength):

    np.save('/Users/guillaumefreychet/Desktop/q.npy', q)
    np.save('/Users/guillaumefreychet/Desktop/profile.npy', profiles)

    I_peaks, q_peaks, wavel = [[]] * 20, [[]] * 20, [[]] * 20
    q_pitch = 2. * np.pi / pitch
    q_first = 6 * q_pitch
    ind = np.argmin(np.abs(q - q_first))
    print(q_pitch, ind)

    #Fit gaussian
    pos_gauss = np.linspace(max(ind - 25, 0), min(ind + 25, len(profiles)-1), int(min(ind + 25, len(profiles)-1) - (max(ind - 25, 0))) - 1, dtype=np.int32)
    g1 = models.Gaussian1D(amplitude = profiles.max(), mean = ind, stddev = 3.)
    g_init = g1
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

    print(g.mean.value, g.amplitude.value)

    Q_peaks = []
    j = 0
    while abs((j+1) * ind_ref) < len(profiles) - 16:
        ind = int((j / 6.) + ind_ref)
        #Switch to masked array
        pos_gauss1 = np.linspace(max(ind - 15, 0), min(ind + 15, len(profiles)), int(1 + min(ind + 15, len(profiles)) - (max(ind - 15, 0))), dtype=np.int32)
        g2 = models.Gaussian1D(amplitude=profiles[pos_gauss1].max(), mean=q[ind], stddev=0.001)
        g_init1 = g2
        fit_g = fitting.LevMarLSQFitter()
        gg = fit_g(g_init1, q[pos_gauss1], profiles[pos_gauss1])

        if gg.mean.value < min(q[pos_gauss1]) or gg.mean.value > max(q[pos_gauss1]):
            Q_peaks.append(q[ind])
            I_peaks[j] = I_peaks[j] + [profiles[pos_gauss1].max()]
        else:
            Q_peaks.append(gg.mean.value)
            I_peaks[j] = I_peaks[j] + [gg.amplitude.value]

        q_peaks[j] = q_peaks[j] + [Q_peaks[j]]
        wavel[j] = wavel[j] + [wavelength]
        j += 1


    return q_peaks, I_peaks, wavel
