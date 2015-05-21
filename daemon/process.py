
import pipeline

from nexpy.api import nexus
import numpy as np
import os
from PIL import Image
import string


def process(paths, experiment,
            options=dict(remesh=False, findcenter=False, refinecenter=False, cachethumbnail=True, variation=True,
                         savefullres=False)):
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
            # if False:  # log image is needed?
            #    with np.errstate(invalid='ignore'):
            #        logimg = np.log(img * (img > 0) + 1)
            thumb = None
            if options['cachethumbnail']:
                thumb = thumbnail(img)

            if options['remesh']:
                img = pipeline.remesh.remesh(img, path, experiment.getGeometry())

            variation = None
            if options['variation']:
                prevpath = previousframe(path)
                if prevpath is not None:
                    print 'comparing:', prevpath, path
                    previmg, _ = pipeline.loader.loadpath(prevpath)
                    variation = pipeline.variation.variation(pipeline.variation.chisquared, img, previmg)
                    print 'variation:', variation
                else:
                    variation = None

            if img is None: return None

            if not options['savefullres']:
                img = None

            outputnexus(img, thumb, path2nexus(path), variation)


def previousframe(path):
    try:
        framenum = os.path.splitext(os.path.basename(path).split('_')[-1])[0]
        prevframenum = int(framenum) - 1
        prevframenum = '{:0>5}'.format(prevframenum)
        return string.replace(path, framenum, prevframenum)
    except ValueError:
        print('No earlier frame found for ' + path)
        return None

def path2nexus(path):
    return os.path.splitext(path)[0] + '.nxs'

def thumbnail(img):
    im = Image.fromarray((img / np.max(img) * 255.).astype(np.uint8), 'L')
    im.thumbnail((128, 128))
    im = np.log(im * (np.asarray(im) > 0) + 1)
    return im


def outputnexus(img, thumb, path, variation=None):
    # print img
    # x, y = np.meshgrid(*(img.shape))
    neximg = nexus.NXdata(img)  #img
    neximg.thumbnail = thumb
    neximg.variation = variation
    nexroot = nexus.NXroot(neximg)
    #print nexroot.tree
    nexroot.save(path)
    return nexroot
