import numpy as np
from scipy.interpolate import LinearNDInterpolator
from astropy.modeling import models, fitting
from astropy.modeling.models import Const1D
from pyFAI import AzimuthalIntegrator,detectors
from scipy.ndimage import filters
from xicam import config
from scipy import signal
from pipeline import loader

def test(wavelength, substratethickness, substrateattenuation, Pitch, Data):
    file, phi = Data[0], Data[1]
    #data = np.flipud(loader.loadimage(file))
    data = loader.loadimage(file)
    img1 = np.rot90(loader.loadimage(file), 1)

    #load meatadata
    #I_0 = loader.loadparas(file)[config.mapHeader('Izero')]
    #phi = loader.loadparas(file)['phi']
    #aq_t = loader.loadparas(file)['Izero']

    q_n, q_x, q_z, I = reduceto1dprofile(data, phi, wavelength)
    I_cor = correc_Iexp(I, substratethickness, substrateattenuation, phi)
    Qxexp, Q__Z, I_peaks = Find_peak(q_n, I_cor, Pitch, phi, wavelength)
    return I_cor, img1, q_x, q_z, Qxexp, Q__Z, I_peaks

def reduceto1dprofile(data, phi, wavelength):
    AI = AzimuthalIntegrator()
    AI = config.activeExperiment.getAI()
    #AI.load('AI')
    #mask = detectors.Pilatus1M().calc_mask()
    center = config.activeExperiment.center
    print(center)
    mask = detectors.Pilatus300k().calc_mask()
    mask[: , int(center[1] - 20) : int(center[1] + 20)] = 10
    cake, q, chi = AI.integrate2d(data, np.shape(data)[0], np.shape(data)[1], mask=mask)
    cakemask, q, chi = AI.integrate2d(np.ones_like(data), np.shape(data)[0], np.shape(data)[1], mask=mask)

    maskedcake = np.ma.masked_array(cake, mask=cakemask <= 0)
    chiprofile = np.ma.average(maskedcake, axis=1)

    kernel = np.zeros_like(chiprofile)
    kernel[0] = 1
    kernel[len(kernel) / 2] = 1

    tiltprofile = filters.convolve1d(kernel, chiprofile, mode='wrap')
    tilt = tiltprofile.argmax() / 1000. * 360

    # correct tilt
    AI.set_rot3(np.deg2rad(tilt))
    cake, q, chi = AI.integrate2d(data, np.shape(data)[0], np.shape(data)[1], mask=mask)
    cakemask, q, chi = AI.integrate2d(np.ones_like(data), np.shape(data)[0], np.shape(data)[1], mask=mask)
    slc = np.vstack([cake[:2,:],cake[-1,:]]) # for other side, index around 500
    profile1=np.sum(slc,axis=0)

    q_n = np.array([q[val] for val in np.where(profile1 > 0)[0]])
    q_x = np.array(q_n * np.cos(phi + 2 * np.arcsin(q_n * wavelength / (4 * np.pi))))
    q_z = np.array(q_n * np.sin(phi + 2 * np.arcsin(q_n * wavelength / (4 * np.pi))))
    intensity = np.array([val for val in profile1 if val > 0])

    return q_n, q_x, q_z, intensity

def Find_peak(q, profiles, pitch, phi, wavelength):

    Qxexp, Q__Z, I_peaks = [[]] * 10, [[]] * 10, [[]] * 10
    q_pitch = (2. * np.pi / (pitch * np.cos(phi)))
    ind = np.argmin(np.abs(q - q_pitch))

    #Fit gaussian
    pos_gauss = np.linspace(ind - 20, ind + 20, 41, dtype=np.int32)
    g1 = models.Gaussian1D(amplitude = profiles.max(), mean = ind, stddev = 1.)
    s1 = Const1D()

    g_init = s1 + g1
    fit_g = fitting.LevMarLSQFitter()
    g = fit_g(g_init, pos_gauss, profiles[pos_gauss])

    ind_ref = g1.mean.value
    q_peaks = []
    j = 0
    while abs((j+1) * ind_ref) < len(profiles) - 15:
        ind = int((j+1) * ind_ref)
        pos_gauss1 = np.linspace(ind - 15, ind + 15, 31, dtype=np.int32)
        g2 = models.Gaussian1D(amplitude=profiles[pos_gauss1].max(), mean=q[ind], stddev=0.01)
        g_init1 = g2
        fit_g = fitting.LevMarLSQFitter()
        gg = fit_g(g_init1, q[pos_gauss1], profiles[pos_gauss1])

        if gg.mean.value < min(q[pos_gauss1]) or gg.mean.value > max(q[pos_gauss1]):
            q_peaks.append(q[ind])
            I_peaks[j] = I_peaks[j] + [profiles[pos_gauss1].max()]
        else:
            q_peaks.append(gg.mean.value)
            I_peaks[j] = I_peaks[j] + [gg.amplitude.value]

        Qxexp[j] = Qxexp[j] + [q_peaks[j] * np.cos(phi + 2 * np.arcsin(q_peaks[j] * wavelength / (4 * np.pi)))]
        Q__Z[j] = Q__Z[j] + [q_peaks[j] * np.sin(phi + 2 * np.arcsin(q_peaks[j] * wavelength / (4 * np.pi)))]

        j += 1

    return Qxexp, Q__Z, I_peaks

# Correction of the footprint and substrate attenuation // Addition of sample size/sample attenuation and polarization
def correc_Iexp(pr, substratethickness, substrateattenuation, phi):
    footprintcorr = 'True'
    abscorr = 'True'
    samplesizecorr = 'False'
    fwhm, sample_size = 1, 1
    for i in range(0, len(pr), 1):
        footprintfactor = np.cos(phi) if footprintcorr else 1
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