import numpy as np


def radialintegrate(imgdata, experiment, mask=None):
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

    #calculate data radial profile
    y, x = np.indices((data.shape))
    r = np.sqrt((x - centerx) ** 2 + (y - centery) ** 2)
    r = r.astype(np.int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel(), (1 - mask).ravel())
    radialprofile = tbin / nr


    #calculate q spacings
    x = np.arange(radialprofile.shape[0])
    theta = np.arctan2(x * experiment.getvalue('Pixel Size X') * 0.000001,
                       experiment.getvalue('Detector Distance'))
    #theta=x*self.config.getfloat('Detector','Pixel Size')*0.000001/self.config.getfloat('Beamline','Detector Distance')
    wavelength = 1.239842 / experiment.getvalue('Energy')
    q = 4 * np.pi / wavelength * np.sin(theta / 2) * .1

    #save integration to file
    f = open("integ.csv", "w")
    for l, z in zip(q, radialprofile):
        f.write(str(l) + "," + str(z) + "\n")
    f.close()

    #remove 0s

    (q, radialprofile) = ([qvalue for qvalue, Ivalue in zip(q, radialprofile) if Ivalue > 0],
                          [Ivalue for qvalue, Ivalue in zip(q, radialprofile) if Ivalue > 0])

    return (q, radialprofile)
