import os
import sys
import logging
import glob
import h5py
import tifffile
from fabio.fabioimage import fabioimage
from fabio import fabioutils
import numpy as np
import fabio, pyFAI
from pyFAI import detectors


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

#TODO: merge bl832h5image with spoth5

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


class ALS832h5image(fabioimage):

    def __init__(self, data=None, header=None):
        super(ALS832h5image, self).__init__(data=data, header=header)
        self.frames = None
        self.currentframe = 0
        self.header = None
        self._h5 = None
        self._dgroup = None
        self._flats = None
        self._darks = None
        self._sinogram = None

    # Context manager for "with" statement compatibility
    def __enter__(self, *arg, **kwarg):
        return self

    def __exit__(self, *arg, **kwarg):
        self.close()

    def _readheader(self, f):
        if self._h5 is not None:
            self.header = dict(self._h5.attrs)
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
        self.currentframe = frame
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
        return len(self.frames)

    @nframes.setter
    def nframes(self, n):
        pass

    def getsinogram(self, idx=None):
        if idx is None: idx = self.data.shape[0] // 2
        self.sinogram = np.vstack([frame for frame in map(lambda x: self._dgroup[self.frames[x]][0, idx],
                                                          range(self.nframes))])
        return self.sinogram

    def __getitem__(self, item):
        s = []
        if not isinstance(item, tuple) or not isinstance(item, list):
            item = (item, )
        for n in range(3):
            if n == 0:
                stop = len(self)
            elif n == 1:
                stop = self.data.shape[0]
            elif n == 2:
                stop = self.data.shape[1]
            if n < len(item) and isinstance(item[n], slice):
                start = item[n].start if item[n].start is not None else 0
                step = item[n].step if item[n].step is not None else 1
                stop = item[n].stop if item[n].stop is not None else stop
            elif n < len(item) and isinstance(item[n], int):
                if item[n] < 0:
                    start, stop, step = stop + item[n], stop + item[n] + 1, 1
                else:
                    start, stop, step = item[n], item[n] + 1, 1
            else:
                start, step = 0, 1

            s.append((start, stop, step))
        for n, i in enumerate(range(s[0][0], s[0][1], s[0][2])):
            _arr = self._dgroup[self.frames[i]][0, slice(*s[1]), slice(*s[2])]
            if n == 0:  # allocate array
                arr = np.empty((len(range(s[0][0], s[0][1], s[0][2])), _arr.shape[0], _arr.shape[1]))
            arr[n] = _arr
        if arr.shape[0] == 1:
            arr = arr[0]
        return arr

    def __len__(self):
        return self.nframes

    def getframe(self, frame=0):
        self.data = self._dgroup[self.frames[frame]][0]
        return self.data

    def next(self):
        if self.currentframe < self.__len__() - 1:
            self.currentframe += 1
        else:
            raise StopIteration
        return self.getframe(self.currentframe)

    def previous(self):
        if self.currentframe > 0:
            self.currentframe -= 1
            return self.getframe(self.currentframe)
        else:
            raise StopIteration

    def close(self):
        self._h5.close()


fabio.openimage.als832h5 = ALS832h5image
fabioutils.FILETYPES['h5'] = ['als832h5']
fabio.openimage.MAGIC_NUMBERS[21]=(b"\x89\x48\x44\x46",'als832h5')


class DXchangeimage(fabioimage):

    def __init__(self, data=None , header=None):
        super(DXchangeimage, self).__init__(data=data, header=header)
        self.currentframe = 0
        self.header = None
        self._h5 = None
        self._dgroup = None
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
            self.header={'foo': 'bar'} #dict(self._h5.attrs)
            # self.header.update(**self._dgroup.attrs)

    def read(self, f, frame=None):
        self.filename = f
        if frame is None:
            frame = 0
        if self._h5 is None:
            self._h5 = h5py.File(self.filename, 'r')
            self._dgroup = self._find_exchange_group(self._h5)
        self.readheader(f)
        self.currentframe = frame
        self.data = self._dgroup['data'][frame]
        return self

    def _find_exchange_group(self, h5object):
        exchange_groups = [key for key in h5object.keys() if 'exchange' in key]
        if len(exchange_groups) > 1:
            raise RuntimeWarning('More than one exchange group found. Will use \'exchange\'\n'
                                 'Need to use logging for this...')
        return h5object[exchange_groups[0]]

    @property
    def flats(self):
        if self._flats is None:
            self._flats = self._dgroup['data_white']
        return self._flats

    @property
    def darks(self):
        if self._darks is None:
            self._darks = self._dgroup['data_dark']
        return self._darks

    @property
    def nframes(self):
        return len(self.frames)

    @nframes.setter
    def nframes(self, n):
        pass

    def getsinogram(self, idx=None):
        if idx is None: idx = self.data.shape[0]//2
        self.sinogram = self._dgroup['data'][:, idx, :]
        return self.sinogram

    def __getitem__(self, item):
        return self._dgroup['data'][item]

    def __len__(self):
        return self._dgroup['data'].shape[0]

    def getframe(self, frame=0):
        self.data = self._dgroup['data'][frame]
        return self.data

    def next(self):
        if self.currentframe < self.__len__() - 1:
            self.currentframe += 1
        else:
            raise StopIteration
        return self.getframe(self.currentframe)

    def previous(self):
        if self.currentframe > 0:
            self.currentframe -= 1
            return self.getframe(self.currentframe)
        else:
            raise StopIteration

    def close(self):
        self._h5.close()


# fabio.openimage.dxchange = DXchangeimage
# fabioutils.FILETYPES['h5'] = ['dxchange']
# fabio.openimage.MAGIC_NUMBERS[21]=(b"\x89\x48\x44\x46",'dxchange')


class TiffStack(object):
    def __init__(self, paths, header=None):
        super(TiffStack, self).__init__()
        if isinstance(paths, list):
            self.frames = paths
        elif os.path.isdir(paths):
            self.frames = sorted(glob.glob(os.path.join(paths, '*.tiff')))
        self.currentframe = 0
        self.header= header

    def __len__(self):
        return len(self.frames)

    def getframe(self, frame=0):
        print self.frames[frame]
        self.data = tifffile.imread(self.frames[frame], memmap=True)
        return self.data

    def close(self):
        pass


# Testing
if __name__ == '__main__':
    from matplotlib.pyplot import imshow, show
    data = fabio.open('/home/lbluque/TestDatasetsLocal/PtC_Secanel_H2Odrop_8000eV_5X_60nmZP_2s_361proj_15_.h5') #20160218_133234_Gyroid_inject_LFPonly.h5')
    # arr = data[-1,:,:] #.getsinogramchunk(slice(0, 512, 1), slice(1000, 1500, 1))
    slc = (slice(None), slice(None, None, 8), slice(None, None, 8))
    # arr = data.__getitem__(slc)
    arr = data.getsinogram(100)
    # print sorted(data.frames, reverse=True)
    print data.darks.shape
    print data.flats.shape
    # imshow(arr, cmap='gray')
    show()