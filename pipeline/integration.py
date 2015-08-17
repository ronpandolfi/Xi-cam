import numpy as np
import debug
from PySide import QtCore
import multiprocessing
import time


def radialintegrate(dimg, cut=None):
    print 'WARNING: Histogram binning method of radial integration is in use.'

    center = dimg.experiment.center

    # print(self.config.maskingmat)
    mask = dimg.experiment.mask

    if mask is None:
        print("No mask defined, creating temporary empty mask.")
        mask = np.zeros_like(dimg.data)
    elif not mask.shape == dimg.data.shape:
        print("Mask dimensions do not match image dimensions. Mask will be ignored until this is corrected.")
        mask = np.zeros_like(dimg.data)

    print 'Mask:', mask.shape
    print 'Image:', dimg.data.shape
    print 'Center:', dimg.experiment.center



    # else:
    #    mask = self.config.maskingmat

    invmask=1-mask

    #mask data
    data = dimg.data * (invmask)

    print 'invmask:', invmask.shape

    if cut is not None:
        invmask *= cut
        data *= cut

    #calculate data radial profile
    y, x = np.indices(data.shape)
    r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
    r = r.astype(np.int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel(), invmask.ravel())
    with np.errstate(divide='ignore', invalid='ignore'):
        radialprofile = tbin / nr

    q = np.arange(radialprofile.shape[0])

    if dimg.experiment.iscalibrated:
        # calculate q spacings
        x = np.arange(radialprofile.shape[0])
        theta = np.arctan2(x * dimg.experiment.getvalue('Pixel Size X'),
                           dimg.experiment.getvalue('Detector Distance'))
        #theta=x*self.config.getfloat('Detector','Pixel Size')*0.000001/self.config.getfloat('Beamline','Detector Distance')
        wavelength = dimg.experiment.getvalue('Wavelength')
        q = 4 * np.pi / wavelength * np.sin(theta / 2) * 1e-10


        # save integration to file
        #f = open("integ.csv", "w")
        #for l, z in zip(q, radialprofile):
        #    f.write(str(l) + "," + str(z) + "\n")
        #f.close()

        # remove 0s
        # (q, radialprofile) = ([qvalue for qvalue, Ivalue in zip(q, radialprofile) if Ivalue > 0],
        #                      [Ivalue for qvalue, Ivalue in zip(q, radialprofile) if Ivalue > 0])
        radialprofile = radialprofile * (radialprofile > 0) + 0.0001 * (radialprofile <= 0)

        # import hipies.debug
        # hipies.debug.showimage(data)
        # hipies.debug.showimage(invmask)
        # hipies.debug.showimage(r)


    return (q, radialprofile)


def pixel_2Dintegrate(dimg, mask=None):
    centerx = dimg.experiment.getvalue('Center X')
    centery = dimg.experiment.getvalue('Center Y')

    if mask is None:
        print("No mask defined, creating temporary empty mask.")
        mask = np.zeros_like(dimg.data)

    # mask data
    data = dimg.data * (1 - mask)

    # calculate data radial profile
    x, y = np.indices(data.shape)
    r = np.sqrt((x - centerx) ** 2 + (y - centery) ** 2)
    r = r.astype(np.int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel(), (1 - mask).ravel())
    radialprofile = tbin / nr

    return radialprofile


def chi_2Dintegrate(imgdata, cen, mu, mask=None, chires=30):
    """
    Integration over r for a chi range. Output is 30*
    """
    if mask is None:
        print("No mask defined, creating temporary empty mask..")
        mask = np.zeros_like(imgdata)

    # mask data
    data = imgdata * (1 - mask)

    x, y = np.indices(data.shape).astype(np.float)
    r = np.sqrt((x - cen[0]) ** 2 + (y - cen[1]) ** 2)
    r = r.astype(np.int)

    delta = 3

    rinf = mu - delta / 2.
    rsup = mu + delta / 2.

    rmask = ((rinf < r) & (r < rsup) & (x < cen[0])).astype(np.int)
    data *= rmask

    chi = chires * np.arctan((y - cen[1]) / (x - cen[0]))
    chi += chires * np.pi / 2.
    chi = np.round(chi).astype(np.int)
    chi *= (chi > 0)

    tbin = np.bincount(chi.ravel(), data.ravel())
    nr = np.bincount(chi.ravel(), rmask.ravel())
    angleprofile = tbin / nr

    # vimodel = pymodelfit.builtins.GaussianModel()
    # vimodel.mu = np.pi / 2 * 100
    # vimodel.A = np.nanmax(angleprofile)
    # vimodel.fitData(x=np.arange(np.size(angleprofile)), y=angleprofile, weights=angleprofile)
    # vimodel.plot(lower=0, upper=np.pi * 100)
    # print ('len:',len(angleprofile))
    return angleprofile


class IntegrationThread(QtCore.QThread):
    sig_Integration = QtCore.Signal(np.ndarray, np.ndarray)

    def run(self, dimg, cut=None):
        resultQueue = multiprocessing.Queue()

        self.p = multiprocessing.Process(target=radialintegratepyFAI, args=(dimg, cut, resultQueue))
        self.p.start()

        while self.p.is_alive():
            time.sleep(.3)

        # q, radialprofile = radialintegratepyFAI(dimg, cut)
        output = resultQueue.get()
        self.sig_Integration.emit(*output)


    def stop(self):
        self.exiting = True
        print ("thread stop - %s" % self.exiting)

    def __del__(self):
        self.exiting = True
        self.wait()


class IntegrationRunner(QtCore.QObject):
    """
    Runs a job in a separate process and forwards messages from the job to the
    main thread through a pyqtSignal.

    """

    result = QtCore.Signal(list, np.ndarray, np.ndarray)

    def __init__(self, start_signal, dimg, cut=None, color=[255, 255, 255]):
        """
        :param start_signal: the pyqtSignal that starts the job

        """
        super(IntegrationRunner, self).__init__()
        self.dimg = dimg
        self.cut = cut
        self.color = color
        start_signal.connect(self._run)

    def _run(self):
        queue = multiprocessing.Queue()
        p = multiprocessing.Process(target=radialintegratepyFAI, args=(self.dimg, self.cut, queue))
        p.start()
        while p.is_alive():
            time.sleep(.3)
        result = queue.get()
        self.result.emit(self.color, *result)




@debug.timeit
def radialintegratepyFAI(dimg, cut=None, resultQueue=False):


    # print(self.config.maskingmat)
    mask = dimg.experiment.mask

    if mask is not None:
        mask = mask.copy()

    if mask is None:
        print("No mask defined, creating temporary empty mask.")
        mask = np.zeros_like(dimg.data)
    elif not mask.shape == dimg.data.shape:
        print("Mask dimensions do not match image dimensions. Mask will be ignored until this is corrected.")
        mask = np.zeros_like(dimg.data)


    # invmask=1-mask

    #mask data
    #data = dimg.data * (invmask)

    #print 'invmask:',invmask.shape


    if cut is not None:
        mask |= 1 - cut
    #        data *= cut



    AI = dimg.experiment.getAI()
    """:type : pyFAI.AzimuthalIntegrator"""

    xres = 2000
    (q, radialprofile) = AI.integrate1d(dimg.data.T, xres, mask=mask.T, method='lut_ocl')
    # Truncate last 3 points, which typically have very high error?

    radialprofile = np.trim_zeros(radialprofile[:-3], 'b')
    q = q[:len(radialprofile)] / 10.0

    if resultQueue:
        resultQueue.put((q, radialprofile))
    else:
        return q, radialprofile


def cake(imgdata, experiment, mask=None, xres=1000, yres=1000):
    # if mask is None:
    # mask = np.zeros_like(imgdata)
    AI = experiment.getAI()
    """:type : pyFAI.AzimuthalIntegrator"""

    return AI.integrate2d(imgdata.T, xres, yres, mask=mask)


def GetArc(Imagedata, center, radius1, radius2, angle1, angle2):
    mask = np.zeros_like(Imagedata)

    centerx = center[0]
    centery = center[1]
    y, x = np.indices(Imagedata.shape)
    r = np.sqrt((x - centerx) ** 2 + (y - centery) ** 2)
    mask = (r > radius1) & (r < radius2)
    theta = np.arctan2(y - centery, x - centerx) / 2 / np.pi * 360
    mask &= (theta > angle1) & (theta < angle2)

    mask = np.flipud(mask)
    return mask * Imagedata
