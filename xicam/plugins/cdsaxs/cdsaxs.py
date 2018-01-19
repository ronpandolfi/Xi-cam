# --coding: utf-8 --
import numpy as np
from astropy.modeling import models, fitting
from pyFAI import AzimuthalIntegrator,detectors
from scipy.ndimage import filters
from xicam import config
from pipeline import loader
import warnings

def configu(beamline, phi, file):
    """ Garbage function which allow to read the different images from several beamlines (CMS, 733, 11012 .....)
    It allow to read the I0, sample_rotation and correct the noise level => different detector and header (different name and level of noise)

    Parameters
    ----------
    a (string) : Name of the beamline
    file (list of string) : name of the file
    phi (float32) : sample_rotation in radians => since some header do not contained this value, this one corresponds to the one entered by the user

    Returns
    -------
    phi (float32) : sample_rotation in radians (if the angle is contained in the header, the value is overwritten)
    data (array of float32): 2D scattering image
    img1 (array of float32): 2D scattering image rotated, displayed as the raw data
    """
    if beamline == '733':
        data = np.flipud(loader.loadimage(file))
        img1 = np.rot90(loader.loadimage(file), 1)
        I_0 = np.float(loader.loadparas(file)['Izero'])
        phi = np.deg2rad(np.float(loader.loadparas(file)['Sample Rotation Stage']))
        #data = data / np.abs(I_0)
    elif beamline == '1102':
        data = np.rot90(np.flipud(loader.loadimage(file)),1)        #Flipud works for RSOXS data
        img1 = loader.loadimage(file)
        I_0 = loader.loadparas(file)['AI 3 Izero']
        th_sc, th_0 = 0.984, -2.14
        phi = np.deg2rad(th_sc * (loader.loadparas(file)['sample theta'] - th_0) - 90)
        t = loader.loadparas(file)['EXPOSURE']
        data = data / np.abs(I_0 * t)
        data = data/1.
        data -= np.mean(data[ : -5, :])                     #Remove background
        data[np.where(data < 0)] = 0.000001
    else:
        data = np.flipud(loader.loadimage(file))
        img1 = np.rot90(loader.loadimage(file), 1)
    return phi, data, img1


def test(substratethickness, substrateattenuation, Pitch, q_pitch, or_hor, header_dic):
    """ This function is used to manage all the process of extraction from the name of the file (it is not very usefull but was easier to use for multiprocessing). It is just calling the different functions.

    Parameters
    ----------
    substratethickness, substrateattenuation, Pitch (float32): substrate thickness, substrate attenuation, pitch
    Data (list of float32 : ([], [])) : Data contains the list of file sected by the user, as well as the sample rotation values attributed by the User

    Returns
    -------
    I_cor, q_x, q_z (list of float32) : List of all the I/qx/qz values from the 1D profile => used for displaying the carthography
    img1 (2D array of float32) : 2D scattering image => displayed as the raw data
    Qxexp, Q__Z, I_peaks (list of float32) : List of all the qx/qz/I values of peaks => Use as the experimental qx for fitting

    """
    try:
        beamline = 'CMS'
        phi, data, img1 = configu(beamline, header_dic[0], header_dic[1])
        q_n, q_x, q_z, I, wavelength = reduceto1dprofile(data, phi, beamline)
        I_cor = correc_Iexp(I, substratethickness, substrateattenuation, header_dic[0])
        Qxexp, Q__Z, I_peaks = Find_peak(q_n, I_cor, Pitch, header_dic[0], wavelength, q_pitch)
        return I_cor, img1, q_x, q_z, Qxexp, Q__Z, I_peaks

    except IndexError:
        print 'Error in {}'.format(file)

def image_orientation(file):

    data = np.flipud(loader.loadimage(file))
    mask = config.activeExperiment.getDetector().calc_mask()

    AI = AzimuthalIntegrator()
    AI = config.activeExperiment.getAI()
    centerX, directDist, centerY = AI.getFit2D()['centerX'], AI.getFit2D()["directDist"], AI.getFit2D()["centerY"]
    AI.setFit2D(directDist, -25, 21)

    cake, q, chi = AI.integrate2d(data[int(centerY - 20): int(centerY + 20), int(centerX + 25):],
                                  config.settings['Integration Bins (q)'], config.settings['Integration Bins (χ)'])
    cakemask, q, chi = AI.integrate2d(np.ones_like(data[int(centerY - 20): int(centerY + 20), int(centerX + 25):]),
                                      config.settings['Integration Bins (q)'], config.settings['Integration Bins (χ)'])
    maskedcake = np.ma.masked_array(cake, mask=cakemask <= 0)
    chiprofile = np.ma.average(maskedcake, axis=1)

    kernel = np.zeros_like(chiprofile)
    kernel[0] = 1
    kernel[len(kernel) / 2] = 1

    tiltprofile = filters.convolve1d(kernel, chiprofile, mode='wrap')
    tilt = tiltprofile.argmax() / float(config.settings['Integration Bins (χ)']) * 360.

    return tilt

def reduceto1dprofile(data, phi, beamline):
    """ First, q indices are calculated from the 2d image (pixel number). Then a correction of the sample tilt is done (if necessary) through the caking of pyFai. Finally, only few pixels around the scattering peaks coming from the gratings are selected and summed.

    Parameters
    ----------
    data (array of float32): 2D scattering image
    phi (float32) : sample_rotation in radians

    Returns
    -------
    q_n (list of float32) : List of all the q values of the extracted 1D profile
    q_x, q_z, intensity (lists of float32) : List of all the qx/qz/intensity values of the extracted 1D profile => used for displaying the carthography
    wavelength (float32) : wavelength

    """

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

    wavelength = config.activeExperiment.getvalue('Wavelength')

    cake, q, chi = AI.integrate2d(data[int(centerY - 20) : int(centerY + 20), int(centerX + 25):], config.settings['Integration Bins (q)'], config.settings['Integration Bins (χ)'])
    cakemask, q, chi = AI.integrate2d(np.ones_like(data[int(centerY - 20) : int(centerY + 20),  int(centerX + 25):]), config.settings['Integration Bins (q)'], config.settings['Integration Bins (χ)'])
    maskedcake = np.ma.masked_array(cake, mask=cakemask <= 0)
    chiprofile = np.ma.average(maskedcake, axis=1)


    kernel = np.zeros_like(chiprofile)
    kernel[0] = 1
    kernel[len(kernel) / 2] = 1

    tiltprofile = filters.convolve1d(kernel, chiprofile, mode='wrap')

    if beamline == '1102':
        tilt = tiltprofile[0:200].argmax() / float(config.settings['Integration Bins (χ)']) * 360.
        tilt = 0
    else:
        tilt = tiltprofile.argmax() / float(config.settings['Integration Bins (χ)'])* 360.
        #print(tilt)

    #correct tilt
    AI.set_rot3(np.deg2rad(tilt))
    AI.setFit2D(directDist, min(0, centerX), 21)
    cake, q, chi = AI.integrate2d(data[int(centerY - 20) : int(centerY + 20),  int(max(0, centerX)):], config.settings['Integration Bins (q)'], 100)
    cakemask, q, chi = AI.integrate2d(np.ones_like(data[int(centerY - 20) : int(centerY + 20), int(max(0, centerX)):]), config.settings['Integration Bins (q)'], 100)


    slc = np.vstack(cake[45:55,:]) # for other side, index around 500
    profile1=np.sum(slc,axis=0)

    q_n = np.array([q[val] for val in np.where(profile1 > 0)[0]])
    q_x = np.array(q_n * np.cos(phi + 2 * np.arcsin(q_n * wavelength / (4. * np.pi))))
    q_z = np.array(q_n * np.sin(phi + 2 * np.arcsin(q_n * wavelength / (4. * np.pi))))
    intensity = np.array([val for val in profile1 if val > 0])

    return q_n, q_x, q_z, intensity, wavelength


def Find_peak(q, profiles, pitch, phi, wavelength, q_pitch):
    """ Find the peak position and intensity by fitting gaussians

    Parameters
    ----------
    q, profiles (list of float32) : List of all the q/intensity values of the 1D profile
    pitch, phi, wavelength (float32) : pitch, sample_rotation, wavelength

    Returns
    -------
    Qxexp, Q__Z, I_peaks(lists of float32) : List of all the qx/qz/I values of peaks => Use as the experimental qx/qz/I for fitting


    """
    Qxexp, Q__Z, I_peaks = [[]] * 20, [[]] * 20, [[]] * 20
    q_pitch /= (1.* np.cos(phi) + 0.0000001)
    ind_pi = np.argmin(np.abs(q - q_pitch))

    j, q_ref, nb = 0, 0, 0
    while abs((j+1) * ind_pi) < len(profiles)-1:
        ind = (j+1) * ind_pi
        #Switch to masked array
        pos_gauss2 = np.linspace(max(ind - 25, 0), min(ind + 35, len(profiles) - 1), int(1 + min(ind + 35, len(profiles) - 1) - (max(ind - 25, 0))), dtype=np.int32)
        g_init2 = models.Gaussian1D(amplitude=profiles[pos_gauss2].max(), mean=q[ind], stddev=0.01)
        fit_g = fitting.LevMarLSQFitter()
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            # Trigger a warning.
            gg = fit_g(g_init2, q[pos_gauss2], profiles[pos_gauss2])
            I_peaks[j] = I_peaks[j] + [np.float(gg.amplitude.value if (not (w) and not (fit_g.fit_info['param_cov'] is None)) else profiles[pos_gauss2].max())]

            if (not (w) and not (fit_g.fit_info['param_cov'] is None and gg.mean.value < q[pos_gauss2[0]] and gg.mean.value > q[pos_gauss2[-1]])):
                ind_pi = np.int(np.argmin(np.abs(q - gg.mean.value)) / (j+1))
        j += 1

    q_ref = q_pitch
    for j in range(0, len(filter(None, I_peaks)), 1):
        Qxexp[j] = Qxexp[j] + [(j+1) * q_ref * np.cos(phi + 2 * np.arcsin((j+1) * q_ref * wavelength / (4. * np.pi)))]
        Q__Z[j] = Q__Z[j] + [(j+1) * q_ref * np.sin(phi + 2 * np.arcsin((j+1) * q_ref * wavelength / (4. * np.pi)))]

    return Qxexp, Q__Z, I_peaks


# Correction of the footprint and substrate attenuation // Addition of sample size/sample attenuation and polarization
def correc_Iexp(pr, substratethickness, substrateattenuation, phi):
    """ Correction of the intensity on 1d experimental profiles (beam footprint, absorption by the substrate, sample size and beamsize)

    Parameters
    ----------
    pr (list of float32): List of all the intensity values of the extracted 1D profile
    substratethickness, substrateattenuation, phi (float32): substrate thickness, substrate attenuation, sample rotation

    Returns
    -------
    pr (list of float32): List of all the intensity values corrected
    """
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
    """ interpolation of the (qx,qz) carthography. Conversion from 3 arrays (q_x as x, q_z as y and I the intensity at q_x, q_z) to display the cartography

    Parameters
    ----------
    q_x, q_z, I_i (arrays of float32): 1D array of all the qx/qz/I values
    sampling_size (float32): Dimension of the carthography

    Returns
    -------
    img(2D array of float 32): 2D carthography
    """
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