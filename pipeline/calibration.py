from pipeline import center_approx
import integration
import numpy as np
import peakfinding
from xicam import config

from pyFAI import calibrant
from functools import wraps
import msg
from xicam import threads
from scipy import signal


def calibrationAlgorithm(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        msg.showMessage('Calibrating...')
        msg.showBusy()

        runnable = threads.RunnableIterator(f, iterator_args=args, iterator_kwargs=kwargs,
                                            callback_slot=showProgress, finished_slot=calibrationFinish)
        threads.add_to_queue(runnable)

    return wrapped


def calibrationFinish():
    msg.showMessage('Calibration complete!', 4)
    msg.hideBusy()


def showProgress(value):
    msg.showProgress(value)


@calibrationAlgorithm
def fourierAutocorrelation(dimg, calibrantkey):
    yield 0
    if dimg.transformdata is None:
        return
    yield 20

    config.activeExperiment.center = center_approx.center_approx(dimg.transformdata)

    yield 40

    radialprofile = integration.pixel_2Dintegrate(dimg, mask=dimg.mask)

    yield 60

    peaks = np.array(peakfinding.findpeaks(np.arange(len(radialprofile)), radialprofile)).T

    yield 80

    peaks = peaks[peaks[:, 1].argsort()[::-1]]

    for peak in peaks:
        if peak[0] > 25 and not np.isinf(peak[1]):  ####This thresholds the minimum sdd which is acceptable
            bestpeak = peak[0]
            # print peak
            break

    calibrant1stpeak = calibrant.ALL_CALIBRANTS[calibrantkey].dSpacing[0]

    # Calculate sample to detector distance for lowest q peak
    tth = 2 * np.arcsin(0.5 * config.activeExperiment.getvalue('Wavelength') / calibrant1stpeak / 1.e-10)
    tantth = np.tan(tth)
    sdd = bestpeak * config.activeExperiment.getvalue('Pixel Size X') / tantth

    config.activeExperiment.setvalue('Detector Distance', sdd)

    #        self.refinecenter()

    dimg.invalidatecache()

    yield 100


@calibrationAlgorithm
def rickerWavelets(dimg, calibrantkey):
    yield 0

    if dimg.transformdata is None:
        return

    yield 1

    radii = np.arange(30, 100)
    img = dimg.transformdata

    maxval = 0
    center = np.array([0, 0], dtype=np.int)
    for i in range(len(radii)):
        yield int(96. * i / len(radii)) + 1
        w = center_approx.tophat2(radii[i], scale=1000)
        im2 = signal.fftconvolve(img, w, 'same')
        if im2.max() > maxval:
            maxval = im2.max()
            center = np.array(np.unravel_index(im2.argmax(), img.shape))

    config.activeExperiment.center = center

    yield 97

    radialprofile = integration.pixel_2Dintegrate(dimg, mask=dimg.mask)

    yield 98

    peaks = np.array(peakfinding.findpeaks(np.arange(len(radialprofile)), radialprofile)).T

    yield 99

    peaks = peaks[peaks[:, 1].argsort()[::-1]]

    for peak in peaks:
        if peak[0] > 25 and not np.isinf(peak[1]):  ####This thresholds the minimum sdd which is acceptable
            bestpeak = peak[0]
            # print peak
            break

    calibrant1stpeak = calibrant.ALL_CALIBRANTS[calibrantkey].dSpacing[0]

    # Calculate sample to detector distance for lowest q peak
    tth = 2 * np.arcsin(0.5 * config.activeExperiment.getvalue('Wavelength') / calibrant1stpeak / 1.e-10)
    tantth = np.tan(tth)
    sdd = bestpeak * config.activeExperiment.getvalue('Pixel Size X') / tantth

    config.activeExperiment.setvalue('Detector Distance', sdd)

    #        self.refinecenter()

    dimg.invalidatecache()

    yield 100



    # self.replot()
    # self.drawcenter()
