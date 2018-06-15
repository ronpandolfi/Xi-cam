import os
import sys
import inspect
import h5py
import tifffile
import glob
import numpy as np
from fabio.fabioimage import fabioimage
import fabio
import pyFAI
from pyFAI import detectors
import logging
from . import msg
import pyfits
from nexusformat import nexus as nx
from collections import OrderedDict
import re

if fabio._version.MAJOR==0 and fabio._version.MINOR<5:
    from PySide import QtGui
    msgBox = QtGui.QMessageBox()
    msgBox.setText("The FabIO package is an older version and must be updated. Xi-cam can try to install this for you.")
    msgBox.setInformativeText("Would you like to install FabIO?")
    msgBox.setStandardButtons(QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)
    msgBox.setDefaultButton(QtGui.QMessageBox.Yes)

    response = msgBox.exec_()
    if response == QtGui.QMessageBox.Yes:
        import pip
        failure = pip.main(['install','--upgrade','fabio'])

        from xicam.dialogs import infodialog
        if failure:
            infodialog("The FabIO package could not be updated. Please update FabIO manually (i.e. 'pip install --upgrade fabio').",'Update Failed')
        else:
            infodialog("The FabIO package was updated. Please restart Xi-cam.",'Update Success!')
            exit(0)

from fabio import fabioutils, edfimage, tifimage, fabioformats

fabioformats.__dict__['_extension_cache'] = None

h5classes = list()
tiffclasses = list()

def register_tiffclass(cls):
    global tiffclasses
    tiffclasses.append(cls)
    return cls

def register_h5class(cls):
    global h5classes
    h5classes.append(cls)
    return cls


class xicamtiffimage(fabioimage):
    DEFAULT_EXTENTIONS = ['.tiff', '.tif']

    def read(self, filename, frame=None):
        for tiff in tiffclasses:
            if hasattr(tiff,'validate'): # check which class preferably based on the validate staticmethod
                try:
                    tiff.validate(filename, frame)
                except Exception as ex:
                    continue
            try: # if there isn't one, try to read with this class
                return xicamtiffimage._instantiate_read(tiff,filename,frame)
            except Exception as ex:
                continue

        # if custom classes fail, use built-in class
        return xicamtiffimage._instantiate_read(fabio.tifimage.tifimage, filename, frame)

    @staticmethod
    def _instantiate_read(cls,filename,frame):
        fabh5 = cls()
        fabh5.filename = filename
        return fabh5.read(filename, frame)

fabio.openimage.MAGIC_NUMBERS.insert(0,(b"\x49\x49", 'xicamtiff'))


class hdf5image(fabioimage):
    DEFAULT_EXTENTIONS = ['.hdf','.h5','.hdf5']

    # A proxy class which defers to specific hdf5 schema classes
    def read(self, filename, frame=None):
        for h5cls in h5classes:
            if hasattr(h5cls,'validate'): # check which class preferably based on the validate staticmethod
                try:
                    h5cls.validate(filename, frame)
                except Exception as ex:
                    continue
            try: # if there isn't one, try to read with this class
                return hdf5image._instantiate_read(h5cls,filename,frame)
            except Exception as ex:
                continue

    @staticmethod
    def _instantiate_read(cls,filename,frame):
        fabh5 = cls()
        fabh5.filename = filename
        return fabh5.read(filename, frame)

fabio.openimage.MAGIC_NUMBERS.insert(0,(b"\x89\x48\x44\x46", 'hdf5'))


@register_h5class
class ALS832H5image(fabioimage):
    """
    Fabio Image class for ALS Beamline 8.3.2 HDF5 Datasets
    """
    DEFAULT_EXTENTIONS = ['h5']

    def __init__(self, data=None, header=None, transpose=False):
        super(ALS832H5image, self).__init__(data=data, header=header)
        self.frames = None
        self.currentframe = 0
        self.header = None
        self._h5 = None
        self._dgroup = None
        self._flats = None
        self._darks = None

    # Context manager for "with" statement compatibility
    def __enter__(self, *arg, **kwarg):
        return self

    def __exit__(self, *arg, **kwarg):
        self.close()

    def _readheader(self, f):
        if self._h5 is not None:
            self.header = dict(self._h5.attrs)
            self.header.update(**self._dgroup.attrs)

    @staticmethod
    def validate(f, frame=None):
        h5 = h5py.File(f, 'r')
        header = dict(h5.attrs)
        assert header['facility'] == 'als'
        assert header['end_station'] == 'bl832'

    def read(self, f, frame=None):
        self.filename = f
        if frame is None:
            frame = 0
        if self._h5 is None:

            # Check header for unique attributes
            try:
                self._h5 = h5py.File(self.filename, 'r+')
                self._dgroup = self._finddatagroup(self._h5)
                self.readheader(f)
                if self.header['facility'] != 'als' or self.header['end_station'] != 'bl832':
                    raise H5ReadError
            except KeyError:
                raise H5ReadError

            self.frames = [key for key in self._dgroup.keys() if 'bak' not in key and 'drk' not in key]
        dfrm = self._dgroup[self.frames[frame]]
        self.currentframe = frame
        self.data = dfrm[0]
        return self

    def change_dataset_attribute(self, key, value):
        self._dgroup.attrs.modify(key, value)

    def _finddatagroup(self, h5object):
        keys = h5object.keys()
        if len(keys) == 1:
            if isinstance(h5object[keys[0]], h5py.Group):
                group_keys = h5object[keys[0]].keys()
                if isinstance(h5object[keys[0]][group_keys[0]], h5py.Dataset):
                    return h5object[keys[0]]
                else:
                    return self._finddatagroup(h5object[keys[0]])
            else:
                raise H5ReadError('Unable to find dataset group')
        else:
            raise H5ReadError('Unable to find dataset group')

    @property
    def flats(self):
        if self._flats is None:
            self._flats = OrderedDict()
            for key in sorted(self._dgroup.keys()):
                if 'bak' in key:
                    self._flats[key] = self._dgroup[key][0]
        return self._flats

    @property
    def darks(self):
        if self._darks is None:
            if self._darks is None:
                self._darks = OrderedDict()
                for key in sorted(self._dgroup.keys()):
                    if 'drk' in key:
                        self._darks[key] = self._dgroup[key][0]
            return self._darks
        return self._darks

    def flatindices(self):
        i0 = int(self.header['i0cycle'])
        nproj = len(self)
        if i0 > 0:
            indices = list(range(0, nproj, i0))
            if indices[-1] != nproj - 1:
                indices.append(nproj - 1)
        elif i0 == 0:
            indices = [0, nproj - 1]
        return indices

    @property
    def nframes(self):
        return len(self.frames)

    @nframes.setter
    def nframes(self, n):
        pass

    # def getsinogram(self, idx=None):
    #     if idx is None: idx = self.data.shape[0]//2
    #     self.sinogram = np.vstack([frame for frame in map(lambda x: self._dgroup[self.frames[x]][0, idx],
    #                                                               range(self.nframes))])
    #     return self.sinogram

    def __getitem__(self, item):
        s = []
        if not isinstance(item, tuple) and not isinstance(item, list):
            item = (item,)
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
        return np.squeeze(arr)

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


class nexusimage(fabioimage):
    DEFAULT_EXTENTIONS = ['.hdf']

    def read(self, f, frame=None):
        # nxroot = nx.nxload(f)
        # # print nxroot.tree
        # if hasattr(nxroot, 'entry'):
        #     if hasattr(nxroot.entry, 'data'):
        #         if hasattr(nxroot.entry.data,'data'):
        #             self._dgroup = nxroot.entry.data.data

        self.filename = f
        if frame is None:
            frame = 0
        if self._h5 is None:
            # Check header for unique attributes
            self._h5 = h5py.File(self.filename, 'r+')
            self._dgroup = self._h5['entry']['data']['data']
            self.readheader(f)

            self.frames = range(self._dgroup.shape[0])

        dfrm = self._dgroup[self.frames[frame]]
        self.currentframe = frame
        self.data = dfrm



        return self

    @staticmethod
    def validate(f, frame=None):
        h5 = h5py.File(f, 'r')
        assert list(h5.keys())[0] == 'entry'
        assert 'data' in list(h5['entry'].keys())
        assert list(h5['entry']['data'])[0] == 'data'

    def __init__(self, data=None , header=None):
        super(nexusimage, self).__init__(data=data, header=header)
        self.frames = None
        self.currentframe = 0
        self.header = None
        self._h5 = None
        self._dgroup = None
        self._flats = None
        self._darks = None

    # Context manager for "with" statement compatibility
    def __enter__(self, *arg, **kwarg):
        return self

    def change_dataset_attribute(self, key, value):
        self._dgroup.attrs.modify(key, value)

    def __exit__(self, *arg, **kwarg):
        self.close()

    def _readheader(self, f):
        #not really useful at this point
        if self._h5 is not None:
            self.header=dict(self._h5.attrs)
            self.header.update(**self._dgroup.attrs)

    @property
    def flats(self):
        return self._flats

    @property
    def darks(self):
        return self._darks

    def flatindices(self):
        nproj = len(self)
        return [0, nproj - 1]

    @property
    def nframes(self):
        return len(self.frames)

    @nframes.setter
    def nframes(self, n):
        pass

    # def getsinogram(self, idx=None):
    #     if idx is None: idx = self.data.shape[0]//2
    #     self.sinogram = np.vstack([frame for frame in map(lambda x: self._dgroup[self.frames[x]][0, idx],
    #                                                               range(self.nframes))])
    #     return self.sinogram

    def __getitem__(self, item):
        s = []
        if not isinstance(item, tuple) and not isinstance(item, list):
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
            _arr = self._dgroup[self.frames[i]][slice(*s[1]), slice(*s[2])]
            if n == 0:  # allocate array
                arr = np.empty((len(range(s[0][0], s[0][1], s[0][2])), _arr.shape[0], _arr.shape[1]))
            arr[n] = _arr
        if arr.shape[0] == 1:
            arr = arr[0]
        return np.squeeze(arr)

    def __len__(self):
        return self.nframes

    def getframe(self, frame=0):
        self.data = self._dgroup[self.frames[frame]]
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


@register_tiffclass
class tomotifimage(fabioimage):

    """
    Fabio class for tiff images (specifically for tomography)
    """

    DEFAULT_EXTENTIONS = ['.tif', '.tiff']

    def __init__(self, data=None, header=None):
        super(tomotifimage, self).__init__(data=data, header=header)
        self.frames = None
        self.currentframe = 0
        self._dgroup = None
        self.header = None
        self.flats = None
        self.darks = None

    @staticmethod
    def validate(f, frame=None):
        tiff = tifffile.imread(f)
        assert len(tiff.shape) > 2

    def read(self, f, frame=None):
        self._dgroup = tifffile.imread(f)
        self.data = self._dgroup[0]
        self.frames = range(self._dgroup.shape[0])
        return self

    def getframe(self, frame=0):
        self.data = self._dgroup[frame]
        return self.data

    def __getitem__(self, item):
        return self._dgroup[item]

    def __len__(self):
        return self._dgroup.shape[0]

    def flatindices(self):
        nproj = len(self)
        return [0, nproj - 1]

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
        pass

class npyimage(fabioimage):
    DEFAULT_EXTENTIONS = ['.npy']

    def read(self, f, frame=None):
        self.data = np.load(f)
        return self


class hipgisaxsimage(fabioimage):
    DEFAULT_EXTENTIONS = ['out']

    def read(self, f, frame=None):
        data = np.loadtxt(f)
        data = (data / data.max() * ((2 ** 32) - 1)).astype(np.uint32).copy()
        self.data = data
        return self


class fitsimage(fabioimage):
    DESCRIPTION = "FITS file format from astronomy"

    DEFAULT_EXTENTIONS = ["fits"]

    def read(self, f, frame=None):
        self.data = np.rot90(np.fliplr(pyfits.open(f)[2].data), 2)
        return self


class gbimage(fabioimage):
    DEFAULT_EXTENTIONS = ['gb']

    def read(self, f, frame=None):
        data = np.fromfile(f, np.float32)
        if len(data) == 1475 * 1679:
            data.shape = (1679, 1475)
        elif len(data) == 981 * 1043:
            data.shape = (1043, 981)
        elif len(data) == 1475 * 195:
            data.shape = (195, 1475)
        self.data = data
        return self


class rawimage(fabioimage):
    DEFAULT_EXTENTIONS = ['raw']

    def read(self, f, frame=None):
        with open(f, 'r') as f:
            data = np.fromfile(f, dtype=np.int32)
        for name, detector in detectors.ALL_DETECTORS.iteritems():
            if hasattr(detector, 'MAX_SHAPE'):
                # print name, detector.MAX_SHAPE, imgdata.shape[::-1]
                if np.prod(detector.MAX_SHAPE) == len(data):  #
                    detector = detector()
                    msg.logMessage('Detector found: ' + name, msg.INFO)
                    break
            if hasattr(detector, 'BINNED_PIXEL_SIZE'):
                # print detector.BINNED_PIXEL_SIZE.keys()
                if len(data) in [np.prod(np.array(detector.MAX_SHAPE) / b) for b in
                                 detector.BINNED_PIXEL_SIZE.keys()]:
                    detector = detector()
                    msg.logMessage('Detector found with binning: ' + name, msg.INFO)
                    break
        data.shape = detector.MAX_SHAPE
        self.data = data
        return self


# @register_fabioclass
# class H5image(fabioimage):
#     """
#     HDF5 Fabio Image class (hack?) to allow for different internal HDF5 structures.
#     To create a fabimage for another HDF5 structure simply define the class in this module like any other fabimage
#     subclass and include 'H5image' somewhere in its name.
#     """
#
#     # This does not really work because fabio creates the instance of the class with not connection to the filename
#     # and only after instantiation does it call read. Therefore any bypasses at __new__ seem to be futile
#     # def __new__(cls, *args, **kwargs):
#     #     h5image_classes = [image for image in inspect.getmembers(sys.modules[__name__], inspect.isclass)
#     #                        if 'h5' in image[0] and image[0] != 'H5image']
#     #     for image_class in h5image_classes:
#     #         try:
#     #             print 'Testing class ', image_class[0]
#     #             obj = image_class[1](*args, **kwargs)
#     #             # obj.read(obj.filename)
#     #             print 'Success!'
#     #         except Exception:
#     #             continue
#     #         else:
#     #             cls = image_class[1]
#     #             break
#     #     else:
#     #         raise RuntimeError('H5 format not recognized')
#     #     return super(H5image, cls).__new__(cls)
#     #  can call super.__new__ or simply return the instance (obj) and bypass init
#     extensions = ['h5']
#
#     def read(self, filename, frame=None):
#         h5image_classes = [image for image in inspect.getmembers(sys.modules[__name__], inspect.isclass)
#                            if 'H5image' in image[0] and image[0] != 'H5image']
#
#         # this block ensures that the Dxchange class is the last one checked, as it breaks the following loop if
#         # the image does not match this class
#         for image_class in h5image_classes:
#             if 'DXchangeH5image' in str(image_class):
#                 dxchange_class = image_class
#         h5image_classes.remove(dxchange_class)
#         h5image_classes.append(dxchange_class)
#
#         for image_class in h5image_classes:
#             try:
#                 obj = image_class[1](self.data, self.header)
#                 obj.read(filename)
#             except H5ReadError:
#                 # Skip exception and try the next H5 image class
#                 continue
#             else:
#                 # If not error was thrown break out of loop
#                 break
#         else:
#             # If for loop finished without breaking raise ReadError
#             raise H5ReadError('H5 format not recognized')
#         return obj  # return the successfully read object



@register_h5class
class ALS733H5image(fabioimage):
    DEFAULT_EXTENTIONS = ['h5']

    def _readheader(self, f):
        fname = f.name  # get filename from file object
        with h5py.File(fname, 'r') as h:
            self.header = dict(h.attrs)

    def read(self, f, frame=None):
        self.readheader(f)

        # Check header for unique attributes
        try:
            if self.header['facility'] != 'als' or self.header['end_station'] != 'bl733':
                raise H5ReadError
        except KeyError:
            raise H5ReadError

        self.filename = f
        if frame is None:
            frame = 0
        return self.getframe(frame)

    @property
    def nframes(self):
        with h5py.File(self.filename, 'r') as h:
            dset = h[h.keys()[0]]
            ddet = dset[dset.keys()[0]]
            if self.isburst:
                frames = sum(map(lambda key: '.edf' in key, ddet.keys()))
            else:
                frames = 1
        return frames

    def __len__(self):
        return self.nframes

    @nframes.setter
    def nframes(self, n):
        pass

    def getframe(self, frame=None):
        if frame is None:
            frame = 0
        f = self.filename
        with h5py.File(f, 'r') as h:
            dset = h[h.keys()[0]]
            ddet = dset[dset.keys()[0]]
            if self.isburst:
                frames = [key for key in ddet.keys() if '.edf' in key]
                dfrm = ddet[frames[frame]]
            elif self.istiled:
                high = ddet[u'high']
                low = ddet[u'low']
                frames = [high[high.keys()[0]], low[low.keys()[0]]]
                dfrm = frames[frame]
            else:
                dfrm = ddet
            self.data = dfrm[0]
        return self.data

    @property
    def isburst(self):
        try:
            with h5py.File(self.filename, 'r') as h:
                dset = h[h.keys()[0]]
                ddet = dset[dset.keys()[0]]
                return not (u'high' in ddet.keys() and u'low' in ddet.keys())
        except AttributeError:
            return False

    @property
    def istiled(self):
        try:
            with h5py.File(self.filename, 'r') as h:
                dset = h[h.keys()[0]]
                ddet = dset[dset.keys()[0]]
                return u'high' in ddet.keys() and u'low' in ddet.keys()
        except AttributeError:
            return False

# @register_h5class
# class GeneralAPSH5image(fabioimage):
#     """
#     Fabio Image class for arbitrary APS H5 structure
#     """
#     DEFAULT_EXTENTIONS = ['h5']
#     def __init__(self, data=None , header=None):
#         super(GeneralAPSH5image, self).__init__(data=data, header=header)
#         self.frames = None
#         self.currentframe = 0
#         self.header = None
#         self._h5 = None
#         self._dgroup = None
#         self.flats = None
#         self.darks = None
#
#     # Context manager for "with" statement compatibility
#     def __enter__(self, *arg, **kwarg):
#         return self
#
#     def __exit__(self, *arg, **kwarg):
#         self.close()
#
#     def _readheader(self, f):
#         #not really useful at this point
#         if self._h5 is not None:
#             self.header=dict(self._h5.attrs)
#             self.header.update(**self._dgroup.attrs)
#
#     def read(self, f, frame=None):
#         self.filename = f
#         if frame is None:
#             frame = 0
#         if self._h5 is None:
#
#             # Check header for unique attributes
#             self._h5 = h5py.File(self.filename, 'r')
#             self._dgroup = self._finddatagroup(self._h5)
#             self.readheader(f)
#
#             self.frames = range(self._dgroup.shape[0])
#
#         dfrm = self._dgroup[self.frames[frame]]
#         self.currentframe = frame
#         self.data = dfrm
#         return self
#
#     def _finddatagroup(self, h5object):
#         keys = h5object.keys()
#         for key in keys:
#             try:
#                 data, data_key = self._check_if_dataset(h5object, key)
#                 break
#             except TypeError:
#                 pass
#
#         try:
#             return data[data_key]
#         except NameError:
#             raise H5ReadError('Unable to find dataset group')
#
#     def _check_if_dataset(self, h5object, key):
#         #recursively find dataset in h5 tree structure
#         if isinstance(h5object[key], h5py.Dataset):
#             return h5object, key
#         else:
#             try:
#                 for lower_key in h5object[key].keys():
#                     return self._check_if_dataset(h5object[key], lower_key)
#             except AttributeError:
#                 pass
#
#     def flatindices(self):
#         nproj = len(self)
#         return [0, nproj - 1]
#
#     @property
#     def nframes(self):
#         return len(self.frames)
#
#     @nframes.setter
#     def nframes(self, n):
#         pass
#
#     def __getitem__(self, item):
#         s = []
#         if not isinstance(item, tuple) and not isinstance(item, list):
#             item = (item, )
#         for n in range(3):
#             if n == 0:
#                 stop = len(self)
#             elif n == 1:
#                 stop = self.data.shape[0]
#             elif n == 2:
#                 stop = self.data.shape[1]
#             if n < len(item) and isinstance(item[n], slice):
#                     start = item[n].start if item[n].start is not None else 0
#                     step = item[n].step if item[n].step is not None else 1
#                     stop = item[n].stop if item[n].stop is not None else stop
#             elif n < len(item) and isinstance(item[n], int):
#                     if item[n] < 0:
#                         start, stop, step = stop + item[n], stop + item[n] + 1, 1
#                     else:
#                         start, stop, step = item[n], item[n] + 1, 1
#             else:
#                 start, step = 0, 1
#
#             s.append((start, stop, step))
#
#         for n, i in enumerate(range(s[0][0], s[0][1], s[0][2])):
#             _arr = self._dgroup[self.frames[i]][slice(*s[1]), slice(*s[2])]
#             if n == 0:  # allocate array
#                 arr = np.empty((len(range(s[0][0], s[0][1], s[0][2])), _arr.shape[0], _arr.shape[1]))
#             arr[n] = _arr
#         if arr.shape[0] == 1:
#             arr = arr[0]
#         return np.squeeze(arr)
#
#     def __len__(self):
#         return self.nframes
#
#     def getframe(self, frame=0):
#         self.data = self._dgroup[self.frames[frame]]
#         return self.data
#
#     def next(self):
#         if self.currentframe < self.__len__() - 1:
#             self.currentframe += 1
#         else:
#             raise StopIteration
#         return self.getframe(self.currentframe)
#
#     def previous(self):
#         if self.currentframe > 0:
#             self.currentframe -= 1
#             return self.getframe(self.currentframe)
#         else:
#             raise StopIteration
#
#     def close(self):
#         self._h5.close()

@register_h5class
class DXchangeH5image(fabioimage):
    """
    Fabio Image class for Data-Exchange HDF5 Datasets
    """
    DEFAULT_EXTENTIONS = ['h5']
    def __init__(self, data=None, header=None):
        super(DXchangeH5image, self).__init__(data=data, header=header)
        self.currentframe = 0
        self.header = None
        self._h5 = None
        self._dgroup = None
        self._flats = None
        self._darks = None

    # Context manager for "with" statement compatibility
    def __enter__(self, *arg, **kwarg):
        return self

    def __exit__(self, *arg, **kwarg):
        self.close()

    def _readheader(self, f):
        # TODO What data should be read here?
        if self._h5 is not None:
            self.header = {'foo': 'bar'}  # dict(self._h5.attrs)
            # self.header.update(**self._dgroup.attrs)

    def read(self, f, frame=None):
        self.filename = f
        if frame is None:
            frame = 0
        if self._h5 is None:
            self._h5 = h5py.File(self.filename, 'r')
            self._dgroup = self._finddatagroup(self._h5)
        self.readheader(f)
        self.currentframe = frame
        self.nframes = self._dgroup['data'].shape[0]
        self.data = self._dgroup['data'][frame]
        self.frames = range(self._dgroup['data'].shape[0])
        return self

    def _finddatagroup(self, h5object):
        exchange_groups = [key for key in h5object.keys() if 'exchange' in key]
        if len(exchange_groups) > 1:
            raise RuntimeWarning('More than one exchange group found. Will use \'exchange\'\n'
                                 'Need to use logging for this...')
        return h5object[exchange_groups[0]]

    @property
    def flats(self):
        if self._flats is None:
            if 'data_white' in self._dgroup:
                self._flats = self._dgroup['data_white']
        return self._flats

    @property
    def darks(self):
        if self._darks is None:
            if 'data_dark' in self._dgroup:
                self._darks = self._dgroup['data_dark']
        return self._darks

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

    # def flatindices(self):
    #     return None


class TiffStack(object):
    """
    Class for stacking several individual tiffs and viewing as a 3D image in an pyqtgraph ImageView
    """

    def __init__(self, paths, header=None):
        super(TiffStack, self).__init__()
        if isinstance(paths, list):
            self.frames = paths
        elif os.path.isdir(paths):
            self.frames = sorted(glob.glob(os.path.join(paths, '*.tiff')))
        self.currentframe = 0
        self.header= header
        self.rawdata = np.stack([self.getframe(frame) for frame in range(len(self.frames))])

        self._readheader()

    def __len__(self):
        return len(self.frames)

    def getframe(self, frame=0):
        self.data = tifffile.imread(self.frames[frame], memmap=True)
        return self.data

    def _readheader(self):
        #not really useful at this point
        if not self.header:
            self.header = {}
            self.header['shape'] = self.rawdata.shape

    def __getitem__(self, item):
        return self.rawdata[item]

    def close(self):
        pass

class CondensedTiffStack(object):
    """
    Class for 3D tiffs to view in pyqtgraph ImageView - very similar to TiffStack class
    """
    def __init__(self, path, header=None):
        super(CondensedTiffStack, self).__init__()

        self.rawdata = tifffile.imread(path, memmap=True)
        self.frames = range(self.rawdata.shape[0])
        self.header = header
        self._readheader()

    @property
    def classname(self):
        return 'CondensedTiffStack'

    def __len__(self):
        return len(self.frames)

    def _readheader(self):
        #not really useful at this point
        if not self.header:
            self.header = {}
            self.header['shape'] = self.rawdata.shape

    def getframe(self, frame=0):
        return self.rawdata[frame]

    def __getitem__(self, item):
        return self.rawdata[item]

    def close(self):
        pass


class EdfImage(edfimage.EdfImage):
    DEFAULT_EXTENTIONS = ['.edf']

    def read(self, f, frame=None):
        return super(EdfImage, self).read(f, frame)

    def _readheader(self, f):
        super(EdfImage, self)._readheader(f)
        f = f.name.replace('.edf', '.txt')
        if os.path.isfile(f):
            self.header.update(self.scanparas(f))

    @staticmethod
    def scanparas(path):
        if not os.path.isfile(path):
            return dict()

        with open(path, 'r') as f:
            lines = f.readlines()

        paras = OrderedDict()

        # The 7.3.3 txt format is messy, with keyless values, and extra whitespaces

        keylesslines = 0
        for line in lines:
            cells = filter(None, re.split('[=:]+', line))

            key = cells[0].strip()

            if cells.__len__() == 2:
                cells[1] = cells[1].split('/')[0]
                paras[key] = cells[1].strip()
            elif cells.__len__() == 1:
                keylesslines += 1
                paras['Keyless value #' + str(keylesslines)] = key

        return paras



class H5ReadError(IOError):
    """
    Exception class raised when checking for the specific schema/structure of an HDF5 file.
    """
    pass


# # Testing
# if __name__ == '__main__':
#     from matplotlib.pyplot import imshow, show
#     data = fabio.open('/home/lbluque/TestDatasetsLocal/dleucopodia.h5') #20160218_133234_Gyroid_inject_LFPonly.h5')
#     # arr = data[-1,:,:] #.getsinogramchunk(slice(0, 512, 1), slice(1000, 1500, 1))
#     slc = (slice(None), slice(None, None, 8), slice(None, None, 8))
#     # arr = data.__getitem__(slc)
#     arr = data.getsinogram(100)
#     # print sorted(data.frames, reverse=True)
#     # print data.darks.shape
#     # print data.flats.shape
#     imshow(arr, cmap='gray')
#     show()
def tests():

    # print(fabio.open('/home/rp/Downloads/Bar5_sample2_th0.098_10.00s_3451.tif').data)
    print(fabio.open('/home/rp/Downloads/MFLP3_50Al_OK_5TH_Sp2_40549-00001(1).fits').data)

if __name__ == '__main__':

    tests()
