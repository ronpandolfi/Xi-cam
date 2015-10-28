
import pipeline

import numpy as np
import os
from PIL import Image
import string


def process(parent, files, experiment,
            options=dict(remesh=False, findcenter=False, refinecenter=False, cachethumbnail=False, variation=True,
                         savefullres=False)):
    """
    Applies a series of processing steps to a list of files; outputs a nexus file containing all results
    """
    # print('Processing new file: ' + path)
    for f in files:
        path = os.path.join(parent, f)
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
            # logimg = np.log(img * (img > 0) + 1)
            thumb = None
            if options['cachethumbnail']:
                thumb = pipeline.writer.thumbnail(img)

            if options['remesh']:
                img = pipeline.remesh.remesh(img, path, experiment.getGeometry())

            variation = None
            if options['variation']:
                prevpath = pipeline.pathtools.similarframe(path, -1)
                nextpath = pipeline.pathtools.similarframe(path, +1)
                if prevpath is not None and nextpath is not None:
                    # print 'comparing:', prevpath, path, nextpath
                    variation = pipeline.variation.filevariation(1, prevpath, img, nextpath)
                    # print 'variation:', variation
                else:
                    variation = None

            if img is None: return None

            if not options['savefullres']:
                img = None

            pipeline.writer.writenexus(img, thumb, pipeline.pathtools.path2nexus(path), path, variation)


