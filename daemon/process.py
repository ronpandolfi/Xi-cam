
import pipeline

from nexpy.api import nexus
import numpy as np
import os
from PIL import Image


def process(paths, experiment, options=dict(remesh=False, findcenter=True, refinecenter=False, cachethumbnail=True)):
    for path in paths:
        img, _ = pipeline.loader.loadpath(path)
        if img is not None:
            if options['findcenter']:
                cen = pipeline.center_approx.center_approx(img)
                experiment.setvalue('Center X', cen[0])
                experiment.setvalue('Center Y', cen[1])
            if options['refinecenter']:
                pipeline.center_approx.refinecenter(img, experiment)
            logimg = None
            if True:  # log image is needed?
                with np.errstate(invalid='ignore'):
                    logimg = np.log(img * (img > 0) + 1)
            thumb = None
            if options['cachethumbnail']:
                thumb = thumbnail(logimg)

            if options['remesh']:
                img = pipeline.remesh.remesh(img, path, experiment.getGeometry())

            if img is None: return None

            # processing completed, save img
            outputnexus(img, thumb, path2nexus(path))


def path2nexus(path):
    return os.path.splitext(path)[0] + '.nxs'


def thumbnail(img):
    im = Image.fromarray((img / np.max(img) * 255.).astype(np.uint8), 'L')
    im.thumbnail((128, 128))

    return im


def outputnexus(img, thumb, path):
    # print img
    x, y = np.meshgrid(*(img.shape))
    neximg = nexus.NXdata(img, [y, x])
    neximg.thumbnail = thumb
    nexroot = nexus.NXroot(neximg)
    print nexroot.tree
    nexroot.save(path)
    return nexroot