import fabio
import simulation, fitting, cdsaxs
import numpy as np
import multiprocessing
from functools import partial
from collections import OrderedDict
from PySide import QtCore


def readheader(files):
    file = [val for val in files]
    if fabio.open(file[0]).header:
        angle_start, angle_end = 60, 0
        for fi in file:
            angle = fabio.open(fi).header['angle']
            angle_start, angle_end = min(angle_start, angle), max(angle_end, angle)

        angle_step = (angle_end - angle_start + 1) / len(file)
    else:
        angle_start, angle_end, angle_step = -60, -57, 0.5

    return file, angle_start, angle_end, angle_step

def loadRAW(files, Phi_min=-45, Phi_max=45, Phi_step=1, Pitch = 100, substratethickness = 700 * 10 ** -6, substrateattenuation = 200 * 10 ** -6):

    qx, qz, I = [], [], []
    q_pitch = np.abs(2. * np.pi / Pitch)

    #Sort angles and files
    if fabio.open(files[0]).header:
        phi = fabio.open(files[0]).header('angle')
    else:
        phi = [np.deg2rad(Phi_max - i * Phi_step) for i in range(0, 1 + int((Phi_max - Phi_min)/Phi_step), 1)]

    header_dic = OrderedDict(sorted(zip(phi, files), key=lambda t: t[0]))

    #open the closest file with the angle closer to 0 and check it orientation
    key = np.argmin(map(abs, header_dic.keys()))
    tilt = cdsaxs.image_orientation(header_dic[header_dic.keys()[key]])
    #modulo?
    or_hor = abs(0 - tilt) < abs(90 - tilt)
    print(np.rad2deg(tilt), or_hor)

    '''
    #1st peak ....
    '''

    pool = multiprocessing.Pool()
    func = partial(cdsaxs.test, substratethickness, substrateattenuation, Pitch, q_pitch, or_hor)

    I_cor, img1, q_x, q_z, Qxexp, Q__Z, I_peaks = zip(*map(func, header_dic.items()))

    pool.close()

    data = np.stack(img1)
    data = np.log(data - data.min() + 1.)

    I_peaks = [np.array(I_peaks)[:,i] for i in range(len(np.array(I_peaks)[0]))]

    #Adaptative threshold
    threshold = max(map(max, np.array(I_peaks)))[0] /10000.
    column_max = map(max, I_peaks)
    ind = np.where(np.array([item for sublist in column_max for item in sublist]) > threshold)

    for i in ind[0]:
        qx.append(np.array([item for sublist in np.array(Qxexp)[:, i] for item in np.array(sublist)]))
        qz.append(np.array([item for sublist in np.array(Q__Z)[:, i] for item in np.array(sublist)]))
        I.append(np.array([item for sublist in np.array(I_peaks)[i, :] for item in np.array(sublist)]))

    sampling_size = (400, 400)
    qx_carto = np.array([item for sublist in q_x for item in sublist])
    qz_carto = np.array([item for sublist in q_z for item in sublist])
    profiles = np.array([item for sublist in I_cor for item in sublist])

    img = cdsaxs.interpolation(qx_carto, qz_carto, profiles, sampling_size)

    return data, img, qx, qz, I

def fitting_cmaes(qx, qz, I, H=10, LL=20, Beta=70, Num_trap=5, DW=0.11, I0=3, Bkg=1):
    initiale_value = [DW, I0, Bkg, int(H), int(LL)] + [int(Beta) for i in range(0, Num_trap,1)]

    best_corr, best_fitness = fitting.cmaes(data=I, qx=qx, qz=qz, initial_guess=np.asarray(initiale_value), sigma=200, ngen=100, popsize=100, mu=10, N=len(initiale_value), restarts=0, verbose=False, tolhistfun=5e-5, ftarget=None)
    I_fit = SL_model1(qx, qz, best_corr)

    #update_right_widget(Num_trap, best_corr[3], best_corr[4], best_corr[5], best_fitness)
    #update_right_widget(best_corr[3], best_corr[4], best_corr[5:], best_fitness)
    #update_model()
    return I_fit, best_corr[3], best_corr[4], best_corr[5], best_fitness

def fitting_mcmc(self, qx, qz, I, H=10, LL=20, Beta=70, Num_trap=5, DW=0.11, I0=3, Bkg=1):
    initiale_value = [DW, I0, Bkg, int(H), int(LL)] + [int(Beta) for i in range(0, Num_trap, 1)]

    best_corr, best_fitness = fitting.mcmc(data=I, qx=qx, qz=qz, initial_guess=np.asarray(initiale_value),
                                            sigma=200, ngen=200, popsize=100, mu=10, N=len(initiale_value),
                                            restarts=0, verbose=False, tolhistfun=5e-5, ftarget=None)
    Qxfit = SL_model1(qx, qz, best_corr)

    update_right_widget(best_corr[3], best_corr[4], best_corr[5:])
    # self.update_model()

def SL_model1(qx, qz,fit_param):

    DW, I0, Bkg, H, LL, Beta = fit_param[0], fit_param[1], fit_param[2], fit_param[3], fit_param[4], fit_param[5:]

    langle = np.deg2rad(np.asarray(Beta))
    rangle = np.deg2rad(np.asarray(Beta))
    Qxfit = []
    for i in range(len(qz)):
        ff_core = simulation.stacked_trapezoids(qx[i], qz[i], 0, LL, H, langle, rangle)
        Qxfit.append(ff_core)
    Qxfitc = fitting.corrections_DWI0Bk(Qxfit, DW, I0, Bkg, qx, qz)
    return Qxfitc

def update_right_widget(Num_trap, H, LL, Beta, f_val):
    sigDrawParam = QtCore.Signal(object)
    sigDrawParam.emit(H, LL, Beta)


def update_model(self):
    sigDrawModel = QtCore.Signal(object)
    sigDrawModel.emit(self)
