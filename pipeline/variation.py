import numpy as np
import loader
import scipy.ndimage
import warnings
import variationoperators
import msg


def variationiterator(simg,operationindex,roi=None,color=None):
    for i in range(len(simg)):
        yield simg.calcVariation(i, operationindex, roi), color

def scanvariation(filepaths):
    simg = loader.multifilediffimage2(filepaths)
    for t in range(len(simg)):
        variationoperators.chisquared(simg,t,None)

def filevariation(operationindex, filea, c, filec, roi=None):
    p = loader.loadimage(filea)
    # c, _ = loader.loadpath(fileb)
    n = loader.loadimage(filec)
    # print 'previous frame:' + filea
    # print p
    return variation(operationindex, p, c, n, roi)

# Deprecating...
def variation(operationindex, imga, imgb=None, imgc=None, roi=None):
    if imga is not None and imgb is not None and imgc is not None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                # TODO: REFACTOR THIS TO USE THUMBS!!!!!
                p = scipy.ndimage.zoom(imga, 0.1, order=1)
                c = scipy.ndimage.zoom(imgb, 0.1, order=1)
                n = scipy.ndimage.zoom(imgc, 0.1, order=1)
                p = scipy.ndimage.gaussian_filter(p, 3)
                c = scipy.ndimage.gaussian_filter(c, 3)
                n = scipy.ndimage.gaussian_filter(n, 3)
                if roi is not None:
                    roi = scipy.ndimage.zoom(roi, 0.1, order=1)
                    roi = np.flipud(roi)
                else:
                    roi = 1
            with np.errstate(divide='ignore'):
                return variationoperators.operations.values()[operationindex](p, c, n, roi, None, None)
        except TypeError:
            msg.logMessage('Variation could not be determined for a frame.',msg.ERROR)
    else:
        msg.logMessage('Variation could not be determined for a frame.',msg.ERROR)
    return 0