import fabio, pyFAI
import h5py
import numpy as np
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


class bl832h5image(fabioimage):

    def __init__(self, data=None , header=None):
        super(bl832h5image, self).__init__(data=data, header=header)
        self._h5 = None
        self._dgroup = None
        self.frames = None
        self._flats = None
        self._darks = None
        self._sinogram = None

    # Context manager for "with" statement compatibility
    def __enter__(self, *arg, **kwarg):
        return self

    def __exit__(self, *arg, **kwarg):
        self.close()

    def _readheader(self,f):
        if self._h5 is not None:
            self.header=dict(self._h5.attrs)
            self.header.update(**self._dgroup.attrs)

    def read(self, f, frame=None):
        self.filename = f
        if frame is None:
            frame = 0
        if self._h5 is None:
            self._h5 = h5py.File(self.filename, 'r')
            self._dgroup = self._find_dataset_group(self._h5)
            self.frames = [key for key in self._dgroup.keys() if 'bak' not in key and 'drk' not in key]
        self.readheader(f)
        dfrm = self._dgroup[self.frames[frame]]
        self.data = dfrm[0]
        return self

    def _find_dataset_group(self, h5object):
        keys = h5object.keys()
        if len(keys) == 1:
            if isinstance(h5object[keys[0]], h5py.Group):
                group_keys = h5object[keys[0]].keys()
                if isinstance(h5object[keys[0]][group_keys[0]], h5py.Dataset):
                    return h5object[keys[0]]
                else:
                    return self._find_dataset_group(h5object[keys[0]])
            else:
                raise Exception('Unable to find dataset group')
        else:
            raise Exception('Unable to find dataset group')

    @property
    def flats(self):
        if self._flats is None:
            self._flats = np.stack([self._dgroup[key][0] for key in self._dgroup.keys() if 'bak' in key])
        return self._flats

    @property
    def darks(self):
        if self._darks is None:
            self._darks = np.stack([self._dgroup[key][0] for key in self._dgroup.keys() if 'drk' in key])
        return self._darks

    @property
    def nframes(self):
        return sum(map(lambda key: 'bak' not in key and 'drk' not in key, self._dgroup.keys()))

    @nframes.setter
    def nframes(self, n):
        pass

    def getsinogram(self, idx=None):
        if idx is None: idx = self.data.shape[0]//2
        self.sinogram = np.vstack([frame[0, idx] for frame in map(lambda x: self._dgroup[self.frames[x]],
                                                                  range(self.nframes))])
        return self

    def getsinogramchunk(self, proj_slice, sino_slc):
        shape = (proj_slice.stop - proj_slice.start, sino_slc.stop - sino_slc.start, self.data.shape[1])
        arr = np.empty(shape)
        for i in range(proj_slice.start, proj_slice.stop, proj_slice.step):
            arr[i] = self._dgroup[self.frames[i]][0, sino_slc, :]
        return arr

    def __len__(self):
        return self.nframes

    def getframe(self, frame=0):
        self.data = self._dgroup[self.frames[frame]][0]
        return self

    def next(self):
        pass

    def previous(self):
        pass

    def close(self):
        self._h5.close()



fabio.openimage.bl832h5 = bl832h5image
fabioutils.FILETYPES['h5'] = ['bl832h5']
fabio.openimage.MAGIC_NUMBERS[21]=(b"\x89\x48\x44\x46",'bl832h5')

if __name__ == '__main__':
    from matplotlib.pyplot import imshow, show
    data = fabio.open('/home/lbluque/TestDatasetsLocal/dleucopodia.h5') #20160218_133234_Gyroid_inject_LFPonly.h5')
    arr = data.getsinogramchunk(slice(0, 512, 1), slice(1000, 1500, 1))
    print arr.shape
    print data.darks.shape
    print data.flats.shape
    # imshow(data.sinogram, cmap='gray')
    # show()