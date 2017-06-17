import numpy as np
from scipy.interpolate import LinearNDInterpolator
from astropy.modeling import models, fitting

def get_exp_values(qxyi, cut_val):
    print(cut_val)
    delta = 0.001
    dtype = [('qx', np.float32), ('qy', np.float32), ('i', np.float32)]
    Sqxyi = []
    for v in qxyi:
        qx, qy, i, phi = v
        Sqxyi.append((qx, qy, i))
    Qi = np.array(Sqxyi, dtype)
    SQi = np.sort(Qi, order='qy')

    binf, bsup = cut_val - delta, cut_val + delta
    idx = np.where((SQi['qx'] > binf) * (SQi['qx'] < bsup))  # selection contraints by qy vals

    return SQi['i'][idx], SQi['qy'][idx]


def generate_carto(profiles, nb_pixel, Phi_min, Phi_step, pixel_size, sample_detector_distance, wavelength, center_x):
    nv = np.zeros([np.shape(profiles)[1],4], dtype=np.float32)
    QxyiData = np.zeros([np.shape(profiles)[1],4], dtype=np.float32)
    for i in range(0, np.shape(profiles)[0], 1):
        phi = np.radians(Phi_min + i * Phi_step)
        q = [0] * np.shape(profiles)[1]
        qx = [0] * np.shape(profiles)[1]
        qz = [0] * np.shape(profiles)[1]
        for j in range(0, np.shape(profiles)[1], 1):
            q[j] = (2 * np.pi / wavelength) * np.arctan(j * nb_pixel / np.shape(profiles)[1] * pixel_size / sample_detector_distance)
            qx[j] = q[j] * np.cos(phi + 2 * np.arcsin(q[j] * wavelength/ (4 * np.pi)))
            qz[j] = q[j] * np.sin(phi + 2 * np.arcsin(q[j] * wavelength/ (4 * np.pi)))
        nv[:, 0] = qx
        nv[:, 1] = qz
        nv[:, 2] = profiles[i]
        nv[:, 3] = phi
        QxyiData = np.vstack((QxyiData, nv))
    return QxyiData


# Correction of the footprint and substrate attenuation // Addition of sample size/sample attenuation and polarization
def correc_Iexp(Qxyi, substratethickness, substrateattenuation):
    footprintcorr = 'True'
    abscorr = 'True'
    samplesizecorr = 'False'
    fwhm, sample_size = 1, 1
    for i in range(0, len(Qxyi[0]), 1):
        footprintfactor = np.cos(Qxyi[i,3]) if footprintcorr else 1
        absfactor = np.exp(-substratethickness * substrateattenuation * (1 - 1 / (Qxyi[i,3] + 0.000000001))) if abscorr else 1
        Qxyi[i, 2] *= absfactor * footprintfactor
    return Qxyi


def inter_carto(qxyi):
    #Reverse map carthography
    qxyi = qxyi[1167:]
    qy = qxyi[:, 0]
    qz = qxyi[:, 1]
    val = qxyi[:, 2]
    angles = qxyi[:, 3]

    # get angles
    a, i, n = np.unique(angles, return_index=True, return_counts=True)
    inds = dict(zip(a, i))
    if np.sum(n - n[0]) > 0:
        raise ValueError('One of the angles has too many Intensity values')

    # calculate number of columns and rows
    ncol = n[0]
    amax = np.abs(a).max()
    nrow = np.int(ncol * np.sin(np.deg2rad(amax)) + 1)
    nrow = 2 * nrow + 1

    # setup output image
    img = np.zeros((nrow, ncol))
    u, v = np.indices((nrow, ncol))
    u -= nrow // 2

    # temp = np.round(np.rad2deg(np.arctan2(u, v)), decimals=1)
    angle = np.round(2 * np.rad2deg(np.arctan2(u, v))) / 2
    radius = np.round(np.sqrt(u ** 2 + v ** 2)).astype(int)

    it = np.nditer(img, flags=['multi_index'], op_flags=['writeonly'])
    while not it.finished:
        y, x = it.multi_index
        ang = angle[y, x]
        rad = min(radius[y, x], ncol - 1)
        if ang in inds:
            i = inds[ang] + rad
            it[0] = val[i]
        it.iternext()

    return img


def interpolation(qxyi, sampling_size=(400, 400)):
    roi_loc = (sampling_size[0] / 2., sampling_size[1] / 2.)
    roi_size = 400
    img = np.zeros((roi_size, roi_size))

    qj = np.floor(
        ((qxyi[:, 0] - qxyi[:, 0].min()) / (qxyi[:, 0] - qxyi[:, 0].min()).max()) * (sampling_size[0] - 1)).astype(
        np.int32)
    qk = np.floor(((qxyi[:, 1].ravel() - qxyi[:, 1].min()) / (qxyi[:, 1] - qxyi[:, 1].min()).max()) * (
        sampling_size[1] - 1)).astype(np.int32)
    I = qxyi[:, 2].ravel()

    # Area of the cartography to interpolate
    selected_idx = []
    assert qj.size == qk.size, 'uncorrect size for q{x,y} index vector'
    for i in xrange(qj.size):
        if -qk[i] / 2 < qj[i] and qj[i] <= roi_loc[0] + roi_size and roi_loc[1] - roi_size < qk[i] and qk[i] <= roi_loc[1] + roi_size :
            selected_idx.append(i)

    qj_shifted = qj[selected_idx] - qj[selected_idx].min()
    qk_shifted = qk[selected_idx] - qk[selected_idx].min()

    print(qxyi[:, 1].min())
    print(qxyi[:, 1].max())

    Isel = I[selected_idx]
    for i, isel in enumerate(Isel):
        img[qj_shifted[i], qk_shifted[i]] += isel
    '''
    plt.imshow((img))
    plt.show()
    '''
    qk_shift =  qk[selected_idx].min()

    to_fill = []
    to_fill = np.array(np.where(img == 0)).T

    interp_from = np.where(img != 0)
    origin = (roi_size / 2)

    interpolator = LinearNDInterpolator(interp_from, img[interp_from])
    cpt = 0

    #for p in to_fill:
    #    img[p[0], p[1]] += interpolator(p[0], p[1])


    for p in to_fill:
        if abs((p[1] - origin) / 2) >= (p[0]):
            continue
        try:
            img[p[0], p[1]] += interpolator(p[0], p[1])
        except ValueError:
            cpt += 1
            pass

    log_possible = np.where(img!='nan')
    img[log_possible] = np.log(img[log_possible] - img[log_possible].min() + 1.)
    return img, qk_shift


def find_peaks(profile_carto, QxyiData, wavelength, nb_pixel, pixel_size, sample_detector_distance):
    Int1 = np.amax(profile_carto)
    ind1 = np.int(np.where(profile_carto == Int1)[0])
    pos_gauss = np.linspace(ind1 - 20, ind1 + 20, 41, dtype=np.int32)

    g_init = models.Gaussian1D(amplitude=Int1, mean=ind1, stddev=1.)
    fit_g = fitting.LevMarLSQFitter()
    g = fit_g(g_init, pos_gauss, profile_carto[pos_gauss])
    ind1 = g.mean.value
    limit_ampli = 0.001 * Int1

    ind = []
    ind.append(ind1)
    cnt = 0
    i = 2
    finish = 'False'

    # Put this into a function + smarter way to do it
    while ((i + 1) * ind[0] < np.shape(profile_carto)[0]) and (finish != 'True'):
        ind_imp1 = i * ind[0]
        pos_gauss1 = np.linspace(ind_imp1 - 20, ind_imp1 + 20, 41, dtype=np.int32)
        g_init_1 = models.Gaussian1D(amplitude=limit_ampli, mean=ind_imp1, stddev=1.)
        g_1 = fit_g(g_init_1, pos_gauss1, profile_carto[pos_gauss1])
        ind_imp1 = g_1.mean.value
        if g_1.amplitude.value < limit_ampli and (i + 1) * ind[0] < np.shape(profile_carto)[0]:
            ind_imp2 = (i + 1) * ind[0]
            pos_gauss2 = np.linspace(ind_imp2 - 20, ind_imp2 + 20, 41, dtype=np.int32)
            g_init_2 = models.Gaussian1D(amplitude=limit_ampli, mean=ind_imp2, stddev=1.)
            g_2 = fit_g(g_init_2, pos_gauss2, profile_carto[pos_gauss2])
            ind_imp2 = g_2.mean.value

            if g_2.amplitude < limit_ampli:
                finish = 'True'

            else:
                ind.append(ind_imp1)
                ind.append(ind_imp2)
                i = i + 2

        elif g_1.amplitude.value < limit_ampli and (i + 1) * ind[0] > np.shape(profile_carto)[0]:
            finish = 'True'

        else:
            ind.append(ind_imp1)
            i = i + 1

    # print(len(ind))
    q = []
    Qxexp = []
    Q__Z = []
    for i in range(0, len(ind), 1):
        if i != 6:
            q.append((2 * np.pi / wavelength) * np.sin(np.arctan((((ind[i] * 500 / 400) * nb_pixel / 500) * (pixel_size / sample_detector_distance)))))
            a, b = get_exp_values(QxyiData, q[i])
            Qxexp.append(a), Q__Z.append(b)
    return q, Qxexp, Q__Z