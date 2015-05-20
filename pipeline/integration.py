import numpy as np
import pymodelfit
import matplotlib.pyplot as plt



def radialintegrate(imgdata, experiment, mask=None, cut=None):
    centerx = experiment.getvalue('Center X')
    centery = experiment.getvalue('Center Y')

    # TODO: add checks for mask and center
    # print(self.config.maskingmat)
    if mask is None:
        print("No mask defined, creating temporary empty mask.")
        mask = np.zeros_like(imgdata)
    elif not mask.shape == imgdata.shape:
        print("Mask dimensions do not match image dimensions. Mask will be ignored until this is corrected.")
        mask = np.zeros_like(imgdata)



    # else:
    #    mask = self.config.maskingmat

    invmask=1-mask

    #mask data
    data = imgdata * (invmask)

    if cut is not None:
        invmask *= cut
        data *= cut

    #calculate data radial profile
    x, y = np.indices(data.shape)
    r = np.sqrt((x - centerx) ** 2 + (y - centery) ** 2)
    r = r.astype(np.int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel(), invmask.ravel())
    with np.errstate(divide='ignore', invalid='ignore'):
        radialprofile = tbin / nr

    q = np.arange(radialprofile.shape[0])

    if experiment.iscalibrated:
        # calculate q spacings
        x = np.arange(radialprofile.shape[0])
        theta = np.arctan2(x * experiment.getvalue('Pixel Size X'),
                           experiment.getvalue('Detector Distance'))
        #theta=x*self.config.getfloat('Detector','Pixel Size')*0.000001/self.config.getfloat('Beamline','Detector Distance')
        wavelength = experiment.getvalue('Wavelength')
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

    return (q, radialprofile)


def pixel_2Dintegrate(imgdata, cen, mask=None):
    if mask is None:
        print("No mask defined, creating temporary empty mask.")
        mask = np.zeros_like(imgdata)

    # mask data
    data = imgdata * (1 - mask)

    # calculate data radial profile
    x, y = np.indices(data.shape)
    r = np.sqrt((x - cen[0]) ** 2 + (y - cen[1]) ** 2)
    r = r.astype(np.int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel(), (1 - mask).ravel())
    radialprofile = tbin / nr

    return radialprofile


def chi_2Dintegrate(imgdata, cen, mu, delta, mask=None):
    if mask is None:
        print("No mask defined, creating temporary empty mask..")
        mask = np.zeros_like(imgdata)

    # mask data
    data = imgdata * (1 - mask)

    x, y = np.indices(data.shape).astype(np.float)
    r = np.sqrt((x - cen[0]) ** 2 + (y - cen[1]) ** 2)
    r = r.astype(np.int)

    mu = mu[1]

    rinf = mu - delta / 2.
    rsup = mu + delta / 2.

    rmask = ((rinf < r) & (r < rsup) & (x < cen[0])).astype(np.int)
    data *= rmask

    chi = 100. * np.arctan((y - cen[1]) / (x - cen[0]))
    chi += 100. * np.pi / 2.
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

    return angleprofile




def radialintegratepyFAI(imgdata, experiment, mask=None, cut=None):
    data = imgdata.copy()
    AI = experiment.getAI()
    """:type : pyFAI.AzimuthalIntegrator"""
    if mask is None:
        print("No mask defined, creating temporary empty mask.")
        mask = np.zeros_like(data)
    if cut is not None:
        mask *= cut
        data *= cut
    xres = 10000
    (q, radialprofile) = AI.integrate1d(data, xres, mask=mask, method='full_csr')
    # Truncate last 3 points, which typically have very high error?
    q = q[:-3] / 10.0
    radialprofile = radialprofile[:-3]
    return q, radialprofile


def cake(imgdata, experiment, mask=None):
    AI = experiment.getAI()
    """:type : pyFAI.AzimuthalIntegrator"""
    xres = 1000
    yres = 1000
    return AI.integrate2d(imgdata.T, xres, yres, mask=np.zeros(experiment.getDetector().MAX_SHAPE))
