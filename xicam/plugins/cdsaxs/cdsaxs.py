# --coding: utf-8 --
import numpy as np
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

def configu(a, file, phi):
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
        data = data / np.abs(I_0)
        data -= np.mean(data[ : -5, :])                     #Remove background
        data[np.where(data < 0)] = 0.000001
    else:
        data = np.flipud(loader.loadimage(file))
        img1 = np.rot90(loader.loadimage(file), 1)
        I_0 = 1
        angle = phi
        #data = data / np.abs(I_0)
    return angle, data, img1


def test(wavelength, substratethickness, substrateattenuation, Pitch, Data):
    try:
        file, phi = Data[0], Data[1]
        #data = np.flipud(loader.loadimage(file))
        beamline = 'CMS'
        phi, data, img1 = configu(beamline, file, phi)

        #636, 466
        np.save('/Users/guillaumefreychet/Desktop/data.npy', data)
        print(np.rad2deg(phi))

        q_n, q_x, q_z, I = reduceto1dprofile(data, phi, wavelength)
        I_cor = correc_Iexp(I, substratethickness, substrateattenuation, phi)
        np.save('/Users/guillaumefreychet/Desktop/icor.npy', I_cor)

        Qxexp, Q__Z, I_peaks = Find_peak(q_n, I_cor, Pitch, phi, wavelength)

        return I_cor, img1, q_x, q_z, Qxexp, Q__Z, I_peaks

    except IndexError:
        print 'Error in {}'.format(file)

def reduceto1dprofile(data, phi, wavelength):
    np.save('/Users/guillaumefreychet/Desktop/data1.npy', data)
    beamline = 'CMS'
    #cosmic masking
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
    centerX, directDist, centerY = AI.getFit2D()['centerX'], AI.getFit2D()["directDist"], AI.getFit2D()["centerY"]
    AI.setFit2D(directDist, -25, 21)

    cake, q, chi = AI.integrate2d(data[int(centerY - 20) : int(centerY + 20), int(centerX + 25):], config.settings['Integration Bins (q)'], config.settings['Integration Bins (χ)'])
    cakemask, q, chi = AI.integrate2d(np.ones_like(data[int(centerY - 20) : int(centerY + 20),  int(centerX + 25):]), config.settings['Integration Bins (q)'], config.settings['Integration Bins (χ)'])
    np.save('/Users/guillaumefreychet/Desktop/cake.npy', cake)

    maskedcake = np.ma.masked_array(cake, mask=cakemask <= 0)
    chiprofile = np.ma.average(maskedcake, axis=1)

    kernel = np.zeros_like(chiprofile)
    kernel[0] = 1
    kernel[len(kernel) / 2] = 1

    tiltprofile = filters.convolve1d(kernel, chiprofile, mode='wrap')
    np.save('/Users/guillaumefreychet/Desktop/tiltprofile.npy',tiltprofile)
    #tilt = tiltprofile[0 : int(0.2 * config.settings['Integration Bins (χ)'])].argmax() / float(config.settings['Integration Bins (χ)'])* 360.
    tilt = tiltprofile.argmax() / float(config.settings['Integration Bins (χ)'])* 360.

    #correct tilt
    AI.set_rot3(np.deg2rad(tilt))
    AI.setFit2D(directDist, 0, 21)
    cake, q, chi = AI.integrate2d(data[int(centerY - 20) : int(centerY + 20),  int(centerX):], config.settings['Integration Bins (q)'], 100)
    cakemask, q, chi = AI.integrate2d(np.ones_like(data[int(centerY - 20) : int(centerY + 20), int(centerX):]), config.settings['Integration Bins (q)'], 100)
    np.save('/Users/guillaumefreychet/Desktop/cake1.npy', cake)

    slc = np.vstack(cake[45:55,:]) # for other side, index around 500
    profile1=np.sum(slc,axis=0)

    q_n = np.array([q[val] for val in np.where(profile1 > 0)[0]])
    q_x = np.array(q_n * np.cos(phi + 2 * np.arcsin(q_n * wavelength / (4. * np.pi))))
    q_z = np.array(q_n * np.sin(phi + 2 * np.arcsin(q_n * wavelength / (4. * np.pi))))
    intensity = np.array([val for val in profile1 if val > 0])

    return q_n, q_x, q_z, intensity

def Find_peak(q, profiles, pitch, phi, wavelength):

    np.save('/Users/guillaumefreychet/Desktop/q.npy', q)
    np.save('/Users/guillaumefreychet/Desktop/profile.npy', profiles)

    Qxexp, Q__Z, I_peaks = [[]] * 20, [[]] * 20, [[]] * 20
    q_pitch = np.abs(2. * np.pi / (pitch * np.cos(phi)))
    ind = np.argmin(np.abs(q - q_pitch))
    #print(phi, q_pitch, ind)

    #Fit gaussian
    fit_g = fitting.LevMarLSQFitter()
    #Switch to masked array
    pos_gauss = np.linspace(max(ind - 50, 0), min(ind + 50, len(profiles)-1), int(min(ind + 50, len(profiles)-1) - (max(ind - 50, 0))) - 1, dtype=np.int32)
    pos_gauss1 = np.linspace(max(2*ind - 50, 0), min(2*ind + 50, len(profiles)-1), int(min(2*ind + 50, len(profiles)-1) - (max(2*ind - 50, 0))) - 1, dtype=np.int32)

    g_init = models.Gaussian1D(amplitude = profiles[pos_gauss].max(), mean= q_pitch, stddev = 0.001)
    g_init1 = models.Gaussian1D(amplitude = profiles[pos_gauss1].max(), mean= 2 * q_pitch, stddev = 0.001)

    g = fit_g(g_init, q[pos_gauss], profiles[pos_gauss])
    g1 = fit_g(g_init1, q[pos_gauss1], profiles[pos_gauss1])
    q_temp = min([g.mean.value, 0.5 * g1.mean.value, ], key=lambda x: abs(x - q_pitch))
    q_peaks = []
    j, q_ref, nb = 0, 0, 0

    while abs((j+1) * q_temp) < q[-25]:
        ind = np.argmin(np.abs(q - ((j+1) *  q_temp)))
        #Switch to masked array
        pos_gauss2 = np.linspace(max(ind - 25, 0), min(ind + 25, len(profiles)), int(1 + min(ind + 25, len(profiles)) - (max(ind - 25, 0))), dtype=np.int32)
        g_init2 = models.Gaussian1D(amplitude=profiles[pos_gauss2].max(), mean=q[ind], stddev=0.001)

        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            # Trigger a warning.
            gg = fit_g(g_init2, q[pos_gauss2], profiles[pos_gauss2])
            if not(gg.mean.value <= q[pos_gauss2[0]] and gg.mean.value > q[pos_gauss2[-1]]):
                q_ref = q_ref + (int(not(w)) * gg.mean.value)
                nb = nb + (int(not(w)) * (j+1))
            I_peaks[j] = I_peaks[j] + [np.float(gg.amplitude.value if (not(w) and not(gg.mean.value <= q[pos_gauss2[0]] and gg.mean.value > q[pos_gauss2[-1]])) else profiles[ind])]
        j += 1

    q_ref /= (nb + 0.0000001)
    print('q_ref', q_ref, np.rad2deg(phi))
    for j in range(0, len(filter(None, I_peaks)), 1):
        Qxexp[j] = Qxexp[j] + [(j+1) * q_ref * np.cos(phi + 2 * np.arcsin((j+1) * q_ref * wavelength / (4. * np.pi)))]
        Q__Z[j] = Q__Z[j] + [(j+1) * q_ref * np.sin(phi + 2 * np.arcsin((j+1) * q_ref * wavelength / (4. * np.pi)))]

    '''
    if gg.mean.value < min(q[pos_gauss1]) or gg.mean.value > max(q[pos_gauss1]):
        q_peaks.append(q[ind])
        I_peaks[j] = I_peaks[j] + [profiles[ind]]
        print(j, 'BadFit')
    else:
        q_peaks.append(gg.mean.value)
        I_peaks[j] = I_peaks[j] + [gg.amplitude.value]

    Qxexp[j] = Qxexp[j] + [q_peaks[j] * np.cos(phi + 2 * np.arcsin(q_peaks[j] * wavelength / (4. * np.pi)))]
    Q__Z[j] = Q__Z[j] + [q_peaks[j] * np.sin(phi + 2 * np.arcsin(q_peaks[j] * wavelength / (4. * np.pi)))]

    j += 1

        with warnings.catch_warnings(record=True) as w:

        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        # Trigger a warning.
        g = fit_g(g_init, pos_gauss, profiles[pos_gauss])
        ind_ref = max(ind - 50, 0) + g.mean.value
        #print('step1', ind_ref)
        # Verify somethings
        if len(w)>0:
            for x in w:
                if x.category == AstropyUserWarning:
                    ind2 = int(2. * ind)
                    fit_g2 = fitting.LevMarLSQFitter()
                    pos_gauss2 = np.linspace(max(ind2 - 100, 0), min(ind2 + 100, len(profiles) - 1), int(min(ind2 + 100, len(profiles) - 1) - (max(ind2 - 100, 0))) - 1, dtype=np.int32)
                    g_init2 = models.Gaussian1D(amplitude=profiles[pos_gauss2].max(), mean= np.argmax(profiles[pos_gauss2]), stddev=3.)

                    g4 = fit_g2(g_init2, pos_gauss2, profiles[pos_gauss2])
                    ind_ref = 0.5 * (max(ind2 - 100, 0) + g4.mean.value)

        if g.mean.value == 0.0:
            ind2 = int(2. * ind)
            fit_g2 = fitting.LevMarLSQFitter()
            pos_gauss2 = np.linspace(max(ind2 - 100, 0), min(ind2 + 100, len(profiles) - 1),
                                     int(min(ind2 + 100, len(profiles) - 1) - (max(ind2 - 100, 0))) - 1, dtype=np.int32)
            g_init2 = models.Gaussian1D(amplitude=profiles[pos_gauss2].max(), mean=np.argmax(profiles[pos_gauss2]),
                                        stddev=3.)

            g4 = fit_g2(g_init2, pos_gauss2, profiles[pos_gauss2])
            ind_ref = 0.5 * (max(ind2 - 100, 0) + g4.mean.value)



    q_peaks = []
    j = 0
    while abs((j+1) * ind_ref) < len(profiles) - 25:
        ind = int((j+1) *  ind_ref)
        #Switch to masked array
        pos_gauss1 = np.linspace(max(ind - 25, 0), min(ind + 25, len(profiles)), int(1 + min(ind + 25, len(profiles)) - (max(ind - 25, 0))), dtype=np.int32)
        g_init1 = models.Gaussian1D(amplitude=profiles[pos_gauss1].max(), mean=q[ind], stddev=0.001)
        fit_g3 = fitting.LevMarLSQFitter()
        gg = fit_g3(g_init1, q[pos_gauss1], profiles[pos_gauss1])

        if gg.mean.value < min(q[pos_gauss1]) or gg.mean.value > max(q[pos_gauss1]):
            q_peaks.append(q[ind])
            I_peaks[j] = I_peaks[j] + [profiles[ind]]
            print(j, 'BadFit')
        else:
            q_peaks.append(gg.mean.value)
            I_peaks[j] = I_peaks[j] + [gg.amplitude.value]

        Qxexp[j] = Qxexp[j] + [q_peaks[j] * np.cos(phi + 2 * np.arcsin(q_peaks[j] * wavelength / (4. * np.pi)))]
        Q__Z[j] = Q__Z[j] + [q_peaks[j] * np.sin(phi + 2 * np.arcsin(q_peaks[j] * wavelength / (4. * np.pi)))]

        j += 1
    '''
    return Qxexp, Q__Z, I_peaks

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