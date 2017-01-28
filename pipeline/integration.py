import numpy as np
from xicam import config
from PySide import QtCore
import multiprocessing
import time
import pyFAI
import remesh
import msg
#
#
# def radialintegrate(dimg, cut=None):
#     print 'WARNING: Histogram binning method of radial integration is in use.'
#
#     center = dimg.experiment.center
#
#     # print(self.config.maskingmat)
#     mask = dimg.experiment.mask
#
#     if mask is None:
#         print("No mask defined, creating temporary empty mask.")
#         mask = np.zeros_like(dimg.data)
#     elif not mask.shape == dimg.data.shape:
#         print("Mask dimensions do not match image dimensions. Mask will be ignored until this is corrected.")
#         mask = np.zeros_like(dimg.data)
#
#     print 'Mask:', mask.shape
#     print 'Image:', dimg.data.shape
#     print 'Center:', dimg.experiment.center
#
#
#
#     # else:
#     #    mask = self.config.maskingmat
#
#     invmask=1-mask
#
#     #mask data
#     data = dimg.data * (invmask)
#
#     print 'invmask:', invmask.shape
#
#     if cut is not None:
#         invmask *= cut
#         data *= cut
#
#     #calculate data radial profile
#     y, x = np.indices(data.shape)
#     r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
#     r = r.astype(np.int)
#
#     tbin = np.bincount(r.ravel(), data.ravel())
#     nr = np.bincount(r.ravel(), invmask.ravel())
#     with np.errstate(divide='ignore', invalid='ignore'):
#         radialprofile = tbin / nr
#
#     q = np.arange(radialprofile.shape[0])
#
#     if dimg.experiment.iscalibrated:
#         # calculate q spacings
#         x = np.arange(radialprofile.shape[0])
#         theta = np.arctan2(x * dimg.experiment.getvalue('Pixel Size X'),
#                            dimg.experiment.getvalue('Detector Distance'))
#         #theta=x*self.config.getfloat('Detector','Pixel Size')*0.000001/self.config.getfloat('Beamline','Detector Distance')
#         wavelength = dimg.experiment.getvalue('Wavelength')
#         q = 4 * np.pi / wavelength * np.sin(theta / 2) * 1e-10
#
#
#         # save integration to file
#         #f = open("integ.csv", "w")
#         #for l, z in zip(q, radialprofile):
#         #    f.write(str(l) + "," + str(z) + "\n")
#         #f.close()
#
#         # remove 0s
#         # (q, radialprofile) = ([qvalue for qvalue, Ivalue in zip(q, radialprofile) if Ivalue > 0],
#         #                      [Ivalue for qvalue, Ivalue in zip(q, radialprofile) if Ivalue > 0])
#         radialprofile = radialprofile * (radialprofile > 0) + 0.0001 * (radialprofile <= 0)
#
#         # import xicam.debug
#         # xicam.debug.showimage(data)
#         # xicam.debug.showimage(invmask)
#         # xicam.debug.showimage(r)
#
#
#     return (q, radialprofile)


def pixel_2Dintegrate(dimg, mask=None):
    centerx = dimg.experiment.getvalue('Center X')
    centery = dimg.experiment.getvalue('Center Y')

    if mask is None:
        msg.logMessage("No mask defined, creating temporary empty mask.",msg.INFO)
        mask = np.zeros_like(dimg.rawdata)

    # mask data
    data = dimg.transformdata * (mask)

    # calculate data radial profile
    x, y = np.indices(data.shape)
    r = np.sqrt((x - centerx) ** 2 + (y - centery) ** 2)
    r = r.astype(np.int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel(), (mask).ravel())
    radialprofile = tbin / nr

    return radialprofile


def chi_2Dintegrate(imgdata, cen, mu, mask=None, chires=30):
    """
    Integration over r for a chi range. Output is 30*
    """
    if mask is None:
        msg.logMessage("No mask defined, creating temporary empty mask..",msg.INFO)
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




#@debugtools.timeit
def radialintegratepyFAI(data, mask=None, AIdict=None, cut=None, color=[255, 255, 255], requestkey = None, qvrt = None, qpar = None):
    if mask is None: mask = config.activeExperiment.mask
    if AIdict is None:
        AI = config.activeExperiment.getAI()
        # p1 = AI.get_poni1()
        # p2 = AI.get_poni2()
        # msg.logMessage(('poni:', p1, p2),msg.DEBUG)
    else:
        AI = pyFAI.AzimuthalIntegrator()
        AI.setPyFAI(**AIdict)


    if mask is not None:
        mask = mask.copy()

    msg.logMessage(('image:', data.shape),msg.DEBUG)
    msg.logMessage(('mask:', mask.shape),msg.DEBUG)

    if not mask.shape == data.shape:
        msg.logMessage("No mask match. Mask will be ignored.",msg.WARNING)
        mask = np.ones_like(data)
        msg.logMessage(('emptymask:', mask.shape),msg.DEBUG)

    if cut is not None and type(cut) is np.ndarray:
        msg.logMessage(('cut:', cut.shape),msg.DEBUG)
        mask = mask.astype(bool) & cut.astype(bool)

    xres = 2000
    (q, radialprofile) = AI.integrate1d(data.T, xres, mask=1 - mask.T, method='lut_ocl')  #pyfai uses 0-valid mask

    q = q/10.

    return q, radialprofile, color, requestkey


def chiintegratepyFAI(data, mask, AIdict, cut=None, color=[255, 255, 255], requestkey = None, qvrt = None, qpar = None, xres=1000, yres=1000):
    AI = pyFAI.AzimuthalIntegrator()
    AI.setPyFAI(**AIdict)
    # Always do mask with 1-valid, 0's excluded

    if mask is not None:
        mask = mask.copy()

    msg.logMessage(('image:', data.shape),msg.DEBUG)
    msg.logMessage(('mask:', mask.shape),msg.DEBUG)

    if not mask.shape == data.shape:
        msg.logMessage("No mask match. Mask will be ignored.",msg.WARNING)
        mask = np.ones_like(data)
        msg.logMessage(('emptymask:', mask.shape),msg.DEBUG)

    if cut is not None:
        msg.logMessage(('cut:', cut.shape),msg.DEBUG)
        mask &= cut.astype(bool)
    # data *= cut


    cake, q, chi = AI.integrate2d(data.T, xres, yres, mask=1 - mask.T, method='lut_ocl')
    mask, q, chi = AI.integrate2d(1 - mask.T, xres, yres, mask=1 - mask.T, method='lut_ocl')

    maskedcake = np.ma.masked_array(cake, mask=mask)

    chiprofile = np.ma.average(maskedcake, axis=1)

    return chi, chiprofile, color, requestkey

def xintegrate(data, mask, AIdict, cut=None, color=[255, 255, 255], requestkey = None, qvrt = None, qpar = None):
    if mask is not None:
        mask = mask.copy()

    msg.logMessage(('image:', data.shape),msg.DEBUG)
    msg.logMessage(('mask:', mask.shape),msg.DEBUG)

    if not mask.shape == data.shape:
        msg.logMessage("No mask match. Mask will be ignored.",msg.WARNING)
        mask = np.ones_like(data)
        msg.logMessage(('emptymask:', mask.shape),msg.INFO)

    if cut is not None:
        msg.logMessage(('cut:', cut.shape),msg.DEBUG)
        mask &= cut.astype(bool)
    # data *= cut


    maskeddata = np.ma.masked_array(data, mask=1-mask)

    xprofile = np.ma.average(maskeddata, axis=1)

    AI = pyFAI.AzimuthalIntegrator()
    AI.setPyFAI(**AIdict)
    qx = AI.qArray(data.shape[::-1])[AI.getFit2D()['centerY'],:]/10
    qx[:qx.argmin()]*=-1

    return qx, xprofile, color, requestkey


def zintegrate(data, mask, AIdict, cut=None, color=[255, 255, 255], requestkey = None, qvrt = None, qpar = None):
    if mask is not None:
        mask = mask.copy()

    msg.logMessage(('image:', data.shape),msg.DEBUG)
    msg.logMessage(('mask:', mask.shape),msg.DEBUG)

    if not mask.shape == data.shape:
        msg.logMessage("No mask match. Mask will be ignored.",msg.WARNING)
        mask = np.ones_like(data)
        msg.logMessage(('emptymask:', mask.shape),msg.DEBUG)

    if cut is not None:
        msg.logMessage(('cut:', cut.shape),msg.DEBUG)
        mask &= cut.astype(bool)
    # data *= cut


    maskeddata = np.ma.masked_array(data, mask=1-mask)

    xprofile = np.ma.average(maskeddata, axis=0)

    AI = pyFAI.AzimuthalIntegrator()
    AI.setPyFAI(**AIdict)
    qz = AI.qArray(data.shape[::-1])[:,AI.getFit2D()['centerX']]/10
    qz[:qz.argmin()]*=-1

    return qz, xprofile, color, requestkey


def cake(imgdata, experiment, mask=None, xres=1000, yres=1000):
    if mask is None:
        mask = np.zeros_like(imgdata)
    AI = experiment.getAI()
    """:type : pyFAI.AzimuthalIntegrator"""

    return AI.integrate2d(imgdata.T, xres, yres, mask=1-mask.T)


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

def qintegrate(*args,**kwargs):
    # if dimg.cakemode:
    #     return xintegrate(dimg.cake, cut, callbackcolor)
    # else:
         return radialintegratepyFAI(*args,**kwargs)

def cakexintegrate(data, mask, AIdict, cut=None, color=[255,255,255], requestkey=None, qvrt = None, qpar = None):
    AI = pyFAI.AzimuthalIntegrator()
    AI.setPyFAI(**AIdict)

    if not mask.shape == data.shape:
        msg.logMessage("No mask match. Mask will be ignored.",msg.WARNING)
        mask = np.ones_like(data)
        msg.logMessage(('emptymask:', mask.shape),msg.DEBUG)

    if cut is not None:
        msg.logMessage(('cut:', cut.shape),msg.DEBUG)
        mask &= cut.astype(bool)

    chi = np.arange(-180,180,360/1000.)

    maskeddata = np.ma.masked_array(data, mask=1-mask)
    xprofile = np.ma.average(maskeddata, axis=1)

    return chi, xprofile, color, requestkey

def cakezintegrate(data, mask, AIdict, cut=None, color=[255,255,255], requestkey=None, qvrt = None, qpar = None):
    AI = pyFAI.AzimuthalIntegrator()
    AI.setPyFAI(**AIdict)

    if not mask.shape == data.shape:
        msg.logMessage("No mask match. Mask will be ignored.",msg.WARNING)
        mask = np.ones_like(data)
        msg.logMessage(('emptymask:', mask.shape),msg.DEBUG)

    if cut is not None:
        msg.logMessage(('cut:', cut.shape),msg.DEBUG)
        mask &= cut.astype(bool)

    q = np.arange(1000)*np.max(qpar)/10000.

    maskeddata = np.ma.masked_array(data, mask=1-mask)
    zprofile = np.ma.average(maskeddata, axis=0)

    return q, zprofile, color, requestkey


def remeshqintegrate(data, mask, AIdict, cut=None, color=[255, 255, 255], requestkey=None, qvrt = None, qpar = None):

    AI = pyFAI.AzimuthalIntegrator()
    AI.setPyFAI(**AIdict)

    alphai=config.activeExperiment.getvalue('Incidence Angle (GIXS)')
    msg.logMessage('Incoming angle applied to remeshed q integration: ' + str(alphai),msg.DEBUG)

    #qpar, qvrt = remesh.remeshqarray(data, None, AI, np.deg2rad(alphai))
    qsquared=qpar**2 + qvrt**2


    remeshcenter=np.unravel_index(qsquared.argmin(),qsquared.shape)

    print 'center?:',remeshcenter

    f2d=AI.getFit2D()
    f2d['centerX']=remeshcenter[0]
    f2d['centerY']=remeshcenter[1]
    AI.setFit2D(**f2d)
    AIdict=AI.getPyFAI()
    msg.logMessage('remesh corrected calibration: '+str(AIdict))

    q,qprofile,color,requestkey = qintegrate(data,mask,AIdict,cut,color,requestkey, qvrt = None, qpar = None)

    maxq = (qsquared*mask).max()
    if cut is not None: qsquared[np.logical_not(cut.astype(np.bool))]=np.inf
    minq = qsquared.min()

    q = np.linspace(minq,maxq,len(qprofile))/10.

    return q, qprofile, color, requestkey

def remeshchiintegrate(data,mask,AIdict,cut=None, color=[255,255,255],requestkey=None, qvrt = None, qpar = None):
    AI = pyFAI.AzimuthalIntegrator()
    AI.setPyFAI(**AIdict)

    alphai=config.activeExperiment.getvalue('Incidence Angle (GIXS)')
    msg.logMessage('Incoming angle applied to remeshed q integration: ' + str(alphai),msg.DEBUG)

    qsquared=qpar**2 + qvrt**2

    remeshcenter=np.unravel_index(qsquared.argmin(),qsquared.shape)

    f2d=AI.getFit2D()
    f2d['centerX']=remeshcenter[0]
    f2d['centerY']=remeshcenter[1]
    AI.setFit2D(**f2d)
    AIdict=AI.getPyFAI()

    return chiintegratepyFAI(data,mask,AIdict,cut,color,requestkey, qvrt = None, qpar = None)

def remeshxintegrate(data, mask, AIdict, cut=None, color=[255, 255, 255], requestkey=None, qvrt = None, qpar = None):
    AI = pyFAI.AzimuthalIntegrator()
    AI.setPyFAI(**AIdict)

    if not mask.shape == data.shape:
        msg.logMessage("No mask match. Mask will be ignored.",msg.WARNING)
        mask = np.ones_like(data)
        msg.logMessage(('emptymask:', mask.shape),msg.DEBUG)

    if cut is not None:
        msg.logMessage(('cut:', cut.shape),msg.DEBUG)
        mask &= cut.astype(bool)

    center = np.where(np.abs(qpar) == np.abs(qpar).min())
    qx = qpar[:,center[0][0]]/10.

    maskeddata = np.ma.masked_array(data, mask=1 - mask)
    xprofile = np.ma.average(maskeddata, axis=1)

    return qx, xprofile, color, requestkey

def remeshzintegrate(data, mask, AIdict, cut=None, color=[255, 255, 255], requestkey=None, qvrt = None, qpar = None):
    AI = pyFAI.AzimuthalIntegrator()
    AI.setPyFAI(**AIdict)

    if not mask.shape == data.shape:
        msg.logMessage("No mask match. Mask will be ignored.",msg.WARNING)
        mask = np.ones_like(data)
        msg.logMessage(('emptymask:', mask.shape),msg.DEBUG)

    if cut is not None:
        msg.logMessage(('cut:', cut.shape),msg.DEBUG)
        mask &= cut.astype(bool)

    center = np.where(np.abs(qvrt) == np.abs(qvrt).min())
    qx = -qvrt[center[1][0]]/10.

    maskeddata = np.ma.masked_array(data, mask=1 - mask)
    xprofile = np.ma.average(maskeddata, axis=0)

    return qx, xprofile, color, requestkey
