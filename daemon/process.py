
import pipeline

from nexpy.api import nexus
import numpy as np
import os
from PIL import Image
import string
from hipies import debug


def process(path, experiment,
            options=dict(remesh=False, findcenter=False, refinecenter=False, cachethumbnail=False, variation=True,
                         savefullres=False)):
    # print('Processing new file: ' + path)

    img, _ = pipeline.loader.loadpath(path)
    if img is not None:
        if options['findcenter']:
            cen = pipeline.center_approx.center_approx(img)
            experiment.setvalue('Center X', cen[0])
            experiment.setvalue('Center Y', cen[1])
        if options['refinecenter']:
            pipeline.center_approx.refinecenter(img, experiment)

        # if False:  # log image is needed?
        # with np.errstate(invalid='ignore'):
        #        logimg = np.log(img * (img > 0) + 1)
        thumb = None
        if options['cachethumbnail']:
            thumb = thumbnail(img)

        if options['remesh']:
            img = pipeline.remesh.remesh(img, path, experiment.getGeometry())

        variation = None
        if options['variation']:
            prevpath = similarframe(path, -1)
            nextpath = similarframe(path, +1)
            if prevpath is not None and nextpath is not None:
                # print 'comparing:', prevpath, path, nextpath
                variation = pipeline.variation.filevariation(4, prevpath, img, nextpath)
                #print 'variation:', variation
            else:
                variation = None

        if img is None: return None

        if not options['savefullres']:
            img = None

        outputnexus(img, thumb, path2nexus(path), path, variation)


def similarframe(path, N):
    try:
        framenum = os.path.splitext(os.path.basename(path).split('_')[-1])[0]
        prevframenum = int(framenum) + N
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


def outputnexus(img, thumb, path, rawpath=None, variation=None):
    # print img
    # x, y = np.meshgrid(*(img.shape))
    neximg = nexus.NXdata(img)  #img
    neximg.rawfile = rawpath
    neximg.thumbnail = thumb
    neximg.variation = variation
    nexroot = nexus.NXroot(neximg)
    #print nexroot.tree
    nexroot.save(path)
    return nexroot
