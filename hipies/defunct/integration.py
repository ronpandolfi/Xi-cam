import numpy as np


def radialintegrate(imgdata, experiment, mask=None, cut=None):
    centerx = experiment.getvalue('Center X')
    centery = experiment.getvalue('Center Y')

    # TODO: add checks for mask and center
    # print(self.config.maskingmat)
    if mask is None:
        print("No mask defined, creating temporary empty mask.")
        mask = np.zeros_like(imgdata)



    #else:
    #    mask = self.config.maskingmat

    #mask data
    data = imgdata * (1 - mask)

    if cut is not None:
        mask *= cut
        data *= cut

    #calculate data radial profile
    x, y = np.indices(data.shape)
    r = np.sqrt((x - centerx) ** 2 + (y - centery) ** 2)
    r = r.astype(np.int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel(), (1 - mask).ravel())
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
        (q, radialprofile) = ([qvalue for qvalue, Ivalue in zip(q, radialprofile) if Ivalue > 0],
                          [Ivalue for qvalue, Ivalue in zip(q, radialprofile) if Ivalue > 0])


    return (q, radialprofile)


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
