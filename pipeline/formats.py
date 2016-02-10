import fabio, pyFAI
from fabio.fabioimage import fabioimage
from fabio import fabioutils
from pyFAI import detectors
import numpy as np
import sys, logging

logger = logging.getLogger("openimage")


class rawimage(fabioimage):
    def read(self, f, frame=None):
        with open(f, 'r') as f:
            data = np.fromfile(f, dtype=np.int32)
        for name, detector in detectors.ALL_DETECTORS.iteritems():
            if hasattr(detector, 'MAX_SHAPE'):
                # print name, detector.MAX_SHAPE, imgdata.shape[::-1]
                if np.prod(detector.MAX_SHAPE) == len(data):  #
                    detector = detector()
                    print 'Detector found: ' + name
                    break
            if hasattr(detector, 'BINNED_PIXEL_SIZE'):
                # print detector.BINNED_PIXEL_SIZE.keys()
                if len(data) in [np.prod(np.array(detector.MAX_SHAPE) / b) for b in
                                 detector.BINNED_PIXEL_SIZE.keys()]:
                    detector = detector()
                    print 'Detector found with binning: ' + name
                    break
        data.shape = detector.MAX_SHAPE
        self.data = data
        return self


fabio.openimage.rawimage = rawimage
fabioutils.FILETYPES['raw'] = ['raw']
from fabio import mar345image

