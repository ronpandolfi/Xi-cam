import fabio, pyFAI
from fabio.fabioimage import fabioimage
from fabio import fabioutils
from pyFAI import detectors
import numpy as np
import sys, logging
import h5py

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

class spoth5image(fabioimage):
    def _readheader(self,f):
        with h5py.File(f,'r') as h:
            self.header=h.attrs
    def read(self,f,frame=None):
        self.filename=f
        if frame is None:
            frame = 0

        return self.getframe(frame)


    @property
    def nframes(self):
        with h5py.File(self.filename,'r') as h:
            dset=h[h.keys()[0]]
            ddet=dset[dset.keys()[0]]
            if self.isburst:
                frames=sum(map(lambda key:'.edf' in key,ddet.keys()))
            else:
                frames = 1
        return frames

    def __len__(self):
        return self.nframes

    @nframes.setter
    def nframes(self,n):
        pass

    def getframe(self,frame=None):
        if frame is None:
            frame = 0
        f = self.filename
        with h5py.File(f,'r') as h:
            dset=h[h.keys()[0]]
            ddet=dset[dset.keys()[0]]
            if self.isburst:
                frames = [key for key in ddet.keys() if '.edf' in key]
                dfrm=ddet[frames[frame]]
            elif self.istiled:
                high = ddet[u'high']
                low  = ddet[u'low']
                frames = [high[high.keys()[0]],low[low.keys()[0]]]
                dfrm = frames[frame]
            else:
                dfrm = ddet
            self.data = dfrm[0]
        return self

    @property
    def isburst(self):
        try:
            with h5py.File(self.filename,'r') as h:
                dset=h[h.keys()[0]]
                ddet=dset[dset.keys()[0]]
                return not (u'high' in ddet.keys() and u'low' in ddet.keys())
        except AttributeError:
            return False

    @property
    def istiled(self):
        try:
            with h5py.File(self.filename,'r') as h:
                dset=h[h.keys()[0]]
                ddet=dset[dset.keys()[0]]
                return u'high' in ddet.keys() and u'low' in ddet.keys()
        except AttributeError:
            return False


fabio.openimage.spoth5image = spoth5image
fabioutils.FILETYPES['h5'] = ['spoth5']
fabio.openimage.MAGIC_NUMBERS[21]=(b"\x89\x48\x44\x46",'spoth5')
