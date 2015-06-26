from nexpy.api import nexus
import numpy as np
import scipy.ndimage
import debug


@debug.timeit
def writenexus(img=None, thumb=None, path=None, rawpath=None, variation=None):
    """
    Output all results to a nexus files
    """

    # x, y = np.meshgrid(*(img.shape))
    neximg = nexus.NXdata(img)
    neximg.rawfile = rawpath
    neximg.thumbnail = thumb
    neximg.variation = variation.items()
    nexroot = nexus.NXroot(neximg)
    # print nexroot.tree
    nexroot.save(path)
    return nexroot


def thumbnail(img, size=160.):
    """
    Generate a thumbnail from an image
    """
    size = float(size)
    img = np.log(img * (img > 0) + 1.)
    img *= 255 / np.max(np.asarray(img))

    desiredsize = np.array([size, size])

    zoomfactor = np.max(desiredsize / np.array(img.shape))
    img = scipy.ndimage.zoom(img, zoomfactor, order=1)
    img = img.astype(np.uint8)
    return img