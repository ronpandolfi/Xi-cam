# -*- coding: UTF-8 -*-

import fabio
import numpy
import pyfits
import os
import numpy as np
#from nexpy.api import nexus as nx
from copy import copy

import detectors
import pyFAI
import glob
import re
import time
import scipy.ndimage
import writer
#import nexpy.api.nexus.tree as tree
from xicam import debugtools, config
from PySide import QtGui
from collections import OrderedDict
# import libtiff

import numpy as nx

import formats  # injects fabio with custom formats


acceptableexts = ['.fits', '.edf', '.tif', '.tiff', '.nxs', '.hdf', '.cbf', '.img', '.raw', '.mar3450', '.gb', '.h5',
                  '.out', '.txt', '.npy']
imagecache = dict()


def loadsingle(path):
    return loadimage(path), loadparas(path)


def loadimage(path):


    data = None
    try:
        if ':' in path:
            print 'WARNING: colon (":") character detected in path; using hack to bypass fabio bug'
            print 'Note: fabio 0.3.0 may have fixed this; check back later'
            path='HACK:'+path
        ext = os.path.splitext(path)[1]
        if ext in acceptableexts:
            if ext == '.gb':
                data = np.fromfile(path, np.float32)
                if len(data) == 1475 * 1679:
                    data.shape = (1679, 1475)
                elif len(data) == 981 * 1043:
                    data.shape = (1043, 981)
                elif len(data) == 1475 * 195:
                    data.shape = (195, 1475)
                return data


            elif ext == '.fits':
                data = np.fliplr(pyfits.open(path)[2].data)
                return data
            elif ext in ['.nxs', '.hdf']:
                nxroot = nx.load(path)
                # print nxroot.tree
                if hasattr(nxroot, 'data'):
                    if hasattr(nxroot.data, 'signal'):
                        data = nxroot.data.signal
                        return data
                    else:
                        return loadimage(str(nxroot.data.rawfile))
            elif ext == '.out':
                data = np.loadtxt(path)
                data = (data / data.max() * ((2 ** 32) - 1)).astype(np.uint32).copy()
                return data
            elif ext == '.npy':
                return np.load( path)
            else:
                data = fabio.open(path).data
                return data
    except IOError:
        print('IO Error loading: ' + path)

    return data


def readenergy(path):
    try:
        if os.path.splitext(path)[1] in acceptableexts:
            if os.path.splitext(path)[1] == '.fits':
                head = pyfits.open(path)
                # print head[0].header.keys()
                paras = scanparaslines(str(head[0].header).split('\r'))
                #print paras
            elif os.path.splitext(path)[1] in ['.nxs', '.hdf']:
                pass
                # nxroot = nx.load(path)
                # # print nxroot.tree
                # if hasattr(nxroot.data, 'signal'):
                # data = nxroot.data.signal
                #     return data, nxroot
                # else:
                #     print ('here:', nxroot.data.rawfile)
                #     return loadsingle(str(nxroot.data.rawfile))
                # print('here',data)

            else:
                pass
    except IOError:
        print('IO Error reading energy: ' + path)

    return None



def readvariation(path):
    for i in range(20):
        try:
            nxroot = nx.load(path)
            print 'Attempt', i + 1, 'to read', path, 'succeeded; continuing...'
            return dict([[int(index), int(value)] for index, value in nxroot.data.variation])
        except IOError:
            print 'Could not load', path, ', trying again in 0.2 s'
            time.sleep(0.2)
        except nx.NeXusError:
            print 'No variation saved in file ' + path

    return None


def merge_dicts(*dict_args):
    '''
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    '''
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


def loadparas(path):
    try:
        extension = os.path.splitext(path)[1]
        if extension == '.fits':
            head = pyfits.open(path)
            # print head[0].header
            return head[0].header

        elif extension == '.edf':
            txtpath = os.path.splitext(path)[0] + '.txt'
            fimg = fabio.open(path)
            if os.path.isfile(txtpath):
                return merge_dicts(fimg.header, scanparas(txtpath))
            else:
                basename = os.path.splitext(path)[0]
                obj = re.search('_\d+$', basename)
                if obj is not None:
                    iend = obj.start()
                    token = basename[:iend] + '*txt'
                    txtpath = glob.glob(token)[0]

                    return merge_dicts(fimg.header, scanparas(txtpath))
                else:
                    return fimg.header

        elif extension == '.gb':
            return {}

        elif extension in ['.nxs', '.hdf']:
            nxroot = nx.load(path)
            # print nxroot.tree
            return nxroot

        elif extension in ['.tif', '.img', '.tiff']:
            fimg = fabio.open(path)
            try:
                frame = int(re.search('\d+(?=.tif)', path).group(0))
                paraspath = re.search('.+(?=_\d+.tif)', path).group(0)
                textheader=scanparas(paraspath, frame)
            except AttributeError:
                textheader=dict()

            return merge_dicts(fimg.header,textheader)

    except IOError:
        print('Unexpected read error in loadparas')
    except IndexError:
        print('No txt file found in loadparas')
    return OrderedDict()


def scanparas(path, frame=None):
    if not os.path.isfile(path):
        return dict()

    with open(path, 'r') as f:
        lines = f.readlines()

    if lines[0][:2] == '#F':
        paras = scanalesandroparaslines(lines, frame)
    else:
        paras = scanparaslines(lines)

    return paras

def scanparaslines(lines):
    paras = OrderedDict()
    for line in lines:
        cells = filter(None, re.split('[=:]+', line))

        key = cells[0]

        if cells.__len__() == 2:
            cells[1] = cells[1].split('/')[0]
            paras[key] = cells[1]
        elif cells.__len__() == 1:
            paras[key] = ''

    return paras


def scanalesandroparaslines(lines, frame):
    keys = []
    values = []
    correctframe = False
    for line in lines:
        token = line[:2]
        if token == '#O':
            keys.extend(line.split()[1:])
        elif token == '#S':
            if int(line.split()[1]) == frame:
                correctframe = True
            else:
                correctframe = False
        elif token == '#P' and correctframe:
            values.extend(line.split()[1:])

    return OrderedDict(zip(keys, values))


def loadstichted(filepath2, filepath1, data1=None, data2=None, paras1=None, paras2=None):
    if data1 is None or data2 is None or paras1 is None or paras2 is None:
        (data1, paras1) = loadsingle(filepath1)
        (data2, paras2) = loadsingle(filepath2)

    # DEFAULT TILING AT 733
    positionY1 = 0 if '_lo_' in filepath1 else 30 * .172
    positionY2 = 30 * .172 if '_hi_' in filepath2 else 0
    positionX1 = 0
    positionX2 = 0

    if 'Detector Vertical' in paras1 and 'Detector Vertical' in paras2 and \
                    'Detector Horizontal' in paras1 and 'Detector Horizontal' in paras2:
        positionY1 = float(paras1['Detector Vertical'])
        positionY2 = float(paras2['Detector Vertical'])
        positionX1 = float(paras1['Detector Horizontal'])
        positionX2 = float(paras2['Detector Horizontal'])

    deltaX = round((positionX2 - positionX1) / 0.172)
    deltaY = round((positionY2 - positionY1) / 0.172)
    padtop2 = 0
    padbottom1 = 0
    padtop1 = 0
    padbottom2 = 0
    padleft2 = 0
    padright1 = 0
    padleft1 = 0
    padright2 = 0

    if deltaY < 0:
        padtop2 = int(abs(deltaY))
        padbottom1 = int(abs(deltaY))
    else:
        padtop1 = int(abs(deltaY))
        padbottom2 = int(abs(deltaY))

    if deltaX < 0:
        padleft2 = int(abs(deltaX))
        padright1 = int(abs(deltaX))

    else:
        padleft1 = int(abs(deltaX))
        padright2 = int(abs(deltaX))

    d2 = numpy.pad(data2, ((padtop2, padbottom2), (padleft2, padright2)), 'constant')
    d1 = numpy.pad(data1, ((padtop1, padbottom1), (padleft1, padright1)), 'constant')

    # mask2 = numpy.pad((data2 > 0), ((padtop2, padbottom2), (padleft2, padright2)), 'constant')
    #mask1 = numpy.pad((data1 > 0), ((padtop1, padbottom1), (padleft1, padright1)), 'constant')
    mask2 = numpy.pad(1 - finddetectorbyfilename(filepath2, data2).calc_mask(),
                      ((padtop2, padbottom2), (padleft2, padright2)),
                      'constant')
    mask1 = numpy.pad(1 - finddetectorbyfilename(filepath1, data1).calc_mask(),
                      ((padtop1, padbottom1), (padleft1, padright1)),
                      'constant')

    with numpy.errstate(divide='ignore'):
        data = (d1 + d2) / (mask2 + mask1)
        data[np.isnan(data)] = 0
    return data


def loadthumbnail(path):
    try:
        if os.path.splitext(path)[1] in ['.nxs', '.hdf']:
            nxroot = nx.load(pathtools.path2nexus(path))
            # print nxroot.tree
            if hasattr(nxroot, 'data'):
                if hasattr(nxroot.data, 'thumbnail'):
                    thumb = np.array(nxroot.data.thumbnail)
                    return thumb
    except IOError:
        print('IO Error loading: ' + path)
    except TypeError:
        print 'TypeError: path has type ', str(type(path))

    return None


# @debugtools.timeit
# def finddetector(imgdata):
#     for name, detector in detectors.ALL_DETECTORS.iteritems():
#         if hasattr(detector, 'MAX_SHAPE'):
#             #print name, detector.MAX_SHAPE, imgdata.shape[::-1]
#             if detector.MAX_SHAPE == imgdata.shape[::-1]:  #
#                 detector = detector()
#                 mask = detector.calc_mask()
#                 print 'Detector found: ' + name
#                 return name, mask, detector
#         if hasattr(detector, 'BINNED_PIXEL_SIZE'):
#             #print detector.BINNED_PIXEL_SIZE.keys()
#             if imgdata.shape[::-1] in [tuple(np.array(detector.MAX_SHAPE) / b) for b in
#                                        detector.BINNED_PIXEL_SIZE.keys()]:
#                 detector = detector()
#                 mask = detector.calc_mask()
#                 print 'Detector found with binning: ' + name
#                 return name, mask, detector
#     raise ValueError('Detector could not be identified!')
#     return None, None, None

def finddetectorbyfilename(path, data=None):
    if data is None:
        data = loadimage(path)

    dimg = diffimage(filepath=path, data=data)
    return dimg.detector


# def loadthumbnail(path):
# # nxpath = pathtools.path2nexus(path)
#
#     img=diffimage(filepath=path).thumbnail
#
#     return img


def loadpath(path):
    if '_lo_' in path or '_hi_' in path:
        try:
            if '_lo_' in path:
                path2 = path.replace('_lo_', '_hi_')
            elif '_hi_' in path:
                path2 = path.replace('_hi_', '_lo_')
            return loadstichted(path, path2)
        except Exception as ex:
            print 'Stitching failed: ', ex.message

    return loadimage(path)


def loadxfs(path):
    return np.loadtxt(path,skiprows=16,converters={0:lambda s: int(s.split(':')[0])*60*60+int(s.split(':')[1])*60+int(s.split(':')[2])})

def convertto8bit(image):
    display_min = image.min()
    display_max = image.max()
    #image = np.array(image, copy=False)
    #image.clip(display_min, display_max, out=image)
    #image -= display_min
    np.true_divide(image,(display_max - display_min + 1) / 256.,out=image,casting='unsafe')
    return image.astype(np.uint8)

def loadtiffstack(path):
    print 'Loading', path +'...'
    data = np.swapaxes(libtiff.TIFF3D.open(path).read_image(),0,1)
    print 'Sub-sampling array...'
    #data = convertto8bit(data)
    print 'Load complete. Size:',np.shape(data)
    data=data[::4,::4,::4]
    return data.astype(np.float32)

def loadimageseries(pattern):
    print 'Loading',pattern+'...'
    files=glob.glob(pattern)
    data=np.dstack([fabio.open(f).data for f in files])
    print 'Log scaling data...'
    data = (np.log(data * (data > 0) + (data < 1)))
    print 'Converting to 8-bit and re-scaling...'
    data = convertto8bit(data)
    print 'Load complete. Size:',np.shape(data)
    return data

import integration, remesh, center_approx, variation, pathtools

class diffimage():
    def __init__(self, filepath=None, data=None, detector=None, experiment=None):
        """
        Image class for diffraction images that caches and validates cache
        :param filepath: str
        :param data: numpy.multiarray.ndarray
        :param detector: pyFAI.detectors.Detector
        :param experiment: xicam.config.experiment
        """

        print 'Loading ' + unicode(filepath) + '...'

        self._data = data

        self.filepath = filepath
        self._detector = detector
        self._params = None
        self._thumb = None
        self._variation = dict()
        self._headers = None
        self._jpeg = None
        self.experiment = experiment
        if self.experiment is None:
            self.experiment = config.activeExperiment





        ### All object I want to save that depend on config parameters must be cached in here instead!!!!
        self.cache = dict()
        self.cachecheck = None

    def updateexperiment(self):
        # Force cache the detector
        _ = self.detector

        # Set experiment energy
        if 'Beamline Energy' in self.params:
            self.experiment.setvalue('Energy', self.params['Beamline Energy'])




    def checkcache(self):
        pass
        # compare experiment with cachecheck

    def invalidatecache(self):
        self.cache = dict()
        print 'cache cleared'

    def cachedata(self):
        if self._data is None:
            if self.filepath is not None:
                try:
                    self._data = loadpath(self.filepath)
                except IOError:
                    debugtools.frustration()
                    raise IOError('File moved, corrupted, or deleted. Load failed')

    @property
    def mask(self):
        if self.experiment.mask is not None:
            return self.experiment.mask
        else:
            return np.ones_like(self.data)

    @property
    def dataunrot(self):
        self.cachedata()
        return self._data

    @property
    def data(self):
        self.cachedata()
        data = self._data
        if self._data is not None:
            data = np.rot90(self._data, 3)
        return data

    @data.setter
    def data(self, data):
        self._data = data

    @property
    def params(self):
        if self._params is None:
            self._params = loadparas(self.paths[0])
        return self._params

    @property
    def detector(self):
        if self._detector is None:
            if self.data is not None:
                name, detector, binning = self.finddetector()
            else:
                return None

            if detector is None:  # If none are found, use the last working one
                return self._detector

            self.detectorname = name
            mask = detector.calc_mask()
            self._detector = detector
            if detector is not None:
                if self.experiment is not None:
                    if mask is not None:
                        self.experiment.addtomask(np.rot90(1 - mask, 3))  # FABIO uses 0-valid mask
                    self.experiment.setvalue('Pixel Size X', detector.pixel1*binning)
                    self.experiment.setvalue('Pixel Size Y', detector.pixel2*binning)
                    self.experiment.setvalue('Detector', name)
        return self._detector

    from fabio import brukerimage

    def finddetector(self):
        for name, detector in sorted(pyFAI.detectors.ALL_DETECTORS.iteritems()):
            print 'det:',name, detector
            if hasattr(detector, 'MAX_SHAPE'):
                # print name, detector.MAX_SHAPE, imgdata.shape[::-1]
                if detector.MAX_SHAPE == self.data.shape[::-1]:  #
                    detector = detector()
                    print 'Detector found: ' + name
                    return name, detector, 1
            if hasattr(detector, 'BINNED_PIXEL_SIZE'):
                # print detector.BINNED_PIXEL_SIZE.keys()
                for binning in detector.BINNED_PIXEL_SIZE.keys():
                    if self.data.shape[::-1] == tuple(np.array(detector.MAX_SHAPE) / binning):
                        detector = detector()
                        print 'Detector found with binning: ' + name
                        return name, detector, binning
        return None, None
        raise ValueError('Detector could not be identified!')

    @detector.setter
    def detector(self, value):
        if type(value) == str:
            try:
                self._detector = pyFAI.detectors.ALL_DETECTORS[value]
            except KeyError:
                try:
                    self._detector = getattr(detectors, value)
                except AttributeError:
                    raise KeyError('Detector not found in pyFAI registry: ' + value)
        else:
            self._detector = value

    @property
    def params(self):
        if self._params is None:
            self._params = loadparas(self.filepath)
        return self._params

    @property
    def thumbnail(self):
        if self._thumb is None:
            self._thumb = loadthumbnail(self.filepath)
            if self._thumb is None:
                self._thumb = writer.thumbnail(self.data)
        return self._thumb

    @property
    def jpeg(self):
        if self._jpeg is None:
            self._jpeg = writer.jpeg(self.data.astype(np.uint8))
        return self._jpeg

    def iscached(self, key):
        return key in self.cache

    def cachedetector(self):
        _=self.detector

    @property
    def cake(self):
        try:
            self.cachedetector()
            if not self.iscached('cake'):
                cake, x, y = integration.cake(self.data, self.experiment)
                cakemask, _, _ = integration.cake(np.ones_like(self.data), self.experiment)
                cakemask = cakemask > 0

                print x, y

                self.cache['cake'] = cake
                self.cache['cakemask'] = cakemask
                self.cache['cakeqx'] = x
                self.cache['cakeqy'] = y

            return self.cache['cake']
        except AttributeError as ex:
            print ex.message

    @property
    def remesh(self):
        if not self.iscached('remesh'):
            print 'headers:', self.headers
            # read incident angle
            if "Sample Alpha Stage" in self.headers:
                alphai = np.deg2rad(float(self.headers["Sample Alpha Stage"]))
            elif "Alpha" in self.headers:
                alphai = np.deg2rad(float(self.headers['Alpha']))
            elif "Sample Theta" in self.headers:
                alphai = np.deg2rad(float(self.headers['Sample Theta']))
            elif "samtilt" in self.headers:
                alphai = np.deg2rad(float(self.headers['samtilt']))
            else:
                alphai = self.queryAlphaI()
            if alphai is None:
                return self.data

            remeshdata, x, y = remesh.remesh(np.rot90(self.data, 1).copy(), self.filepath,
                                             self.experiment.getGeometry(), alphai)
            remeshmask, _, _ = remesh.remesh(np.rot90(self.mask).copy(), self.filepath,
                                             self.experiment.getGeometry(), alphai)

            self.cache['remesh'] = remeshdata
            self.cache['remeshmask'] = remeshmask >0
            self.cache['remeshqx'] = x
            self.cache['remeshqy'] = y

        return self.cache['remesh']

    def queryAlphaI(self):
        alphai, ok = QtGui.QInputDialog.getDouble(None, u'Incident Angle', u'Enter incident angle (degrees):',
                                                  decimals=3)
        if alphai and ok:
            alphai = np.deg2rad(alphai)
            self.headers['Alpha'] = alphai
            return alphai

        return None


    def __del__(self):
        # TODO: do more here!
        # if self._data is not None:
        #    self.writenexus()
        pass

    @debugtools.timeit
    def writenexus(self):
        nxpath = pathtools.path2nexus(self.filepath)
        w = writer.nexusmerger(img=self._data, thumb=self.thumbnail, path=nxpath, rawpath=self.filepath,
                               variation=self._variation)
        w.run()

    def findcenter(self):
        # Auto find the beam center
        [x, y] = center_approx.center_approx(self.data)

        # Set the center in the experiment
        self.experiment.center = (x, y)

    def integrate(self, mode='', cut=None):
        ai = config.activeExperiment.getAI().getPyFAI()
        iscake = False
        return integration.radialintegratepyFAI(self.data, self.mask, ai, cut=cut)

    @debugtools.timeit  #0.07s on Izanami
    def variation(self, operationindex, roi):
        if operationindex not in self._variation or roi is not None:
            nxpath = pathtools.path2nexus(self.filepath)
            if os.path.exists(nxpath) and roi is None:
                v = readvariation(nxpath)
                #print v
                if operationindex in v:
                    self._variation[operationindex] = v[operationindex]
                    print 'successful variation load!'
                else:
                    prv = pathtools.similarframe(self.filepath, -1)
                    nxt = pathtools.similarframe(self.filepath, +1)
                    self._variation[operationindex] = variation.filevariation(operationindex, prv, self.dataunrot, nxt)
            else:
                prv = pathtools.similarframe(self.filepath, -1)
                nxt = pathtools.similarframe(self.filepath, +1)
                if roi is None:
                    print prv, self.dataunrot, nxt
                    self._variation[operationindex] = variation.filevariation(operationindex, prv, self.dataunrot, nxt)
                else:
                    v = variation.filevariation(operationindex, prv, self.dataunrot, nxt, roi)
                    return v
        return self._variation[operationindex]

    @property
    def headers(self):
        if self._headers is None:
            self._headers = loadparas(self.filepath)

        return self._headers

    @property
    def radialintegration(self):
        if 'radialintegration' in self.cache.keys():
            self.cache['radialintegration'] = integration.radialintegrate(self)

        return self.cache['radialintegration']


    def __getattr__(self, name):
       if name in self.cache:
           return self.cache[name]
       else:
           raise AttributeError('diffimage has no attribute: ' + name)


class imageseries():
    def __init__(self, paths, experiment=None):
        self.paths = dict()
        self.variation = dict()
        self.appendimages(paths)
        self.experiment = experiment
        self.roi = None
        self._thumbs = None
        self._dimgs = None
        self._jpegs = None

        if self.experiment is None:
            self.experiment = config.activeExperiment

    @property
    def dimgs(self):
        if self._dimgs is None:
            self._dimgs = [self.__getitem__(i) for i in range(self.__len__())]
        return self._dimgs

    def __len__(self):
        return len(self.paths)

    @property
    def xvals(self):
        return numpy.array(sorted(self.paths.keys()))

    def first(self):
        if len(self.paths) > 0:
            firstpath = sorted(list(self.paths.values()))[0]
            print firstpath
            return diffimage(filepath=firstpath, experiment=self.experiment)
        else:
            return diffimage(data=np.zeros((2, 2)), experiment=self.experiment)

    def __getitem__(self, item):
        return self.getDiffImage(self.paths.keys()[item])

    def getDiffImage(self, key):
        #print self.paths.keys()

        return diffimage(filepath=self.paths[key], experiment=self.experiment)

    def appendimages(self, paths):
        for path in paths:
            frame = self.path2frame(path)
            if frame is None:
                continue

            self.variation[frame] = None
            self.paths[frame] = path

    def currentdiffimage(self):
        pass

    # @property
    # def roi(self):
    # return self._roi
    #
    # @roi.setter
    # def roi(self,value):
    # self._roi=value

    def scan(self, operationindex,roi=None):
        if len(self.paths) < 3:
            return None

        variation = dict()

        # get the first frame's profile
        keys = self.paths.keys()

        if roi is not None:
            roi = writer.thumbnail(roi.T)

        for key, index in zip(keys, range(self.__len__())):
            variationx = self.path2frame(self.paths[key])
            variationy = self.calcVariation(index, operationindex, roi)

            if variationy is not None:
                variation[variationx] = variationy

        return variation

    def calcVariation(self, i, operationindex, roi):
        if roi is None:
            roi = 1
        if i == 0:
            return None  # Prevent wrap-around with first variation

        thumbs = self.thumbs
        try:
            if operationindex == 7:
                return variation.variationoperators.operations.values()[operationindex](self[i].data, i, roi)
            return variation.variationoperators.operations.values()[operationindex](thumbs, i, roi)
        except IndexError as ex:
            print 'Skipping index:', i
        return None

    @property
    @debugtools.timeit
    def thumbs(self):
        if self._thumbs is None:
            self._thumbs = [dimg.thumbnail for dimg in self.dimgs]
        return self._thumbs

    @property
    def jpegs(self):
        if self._jpegs is None:
            self._jpegs = jpegimageset([dimg.jpeg for dimg in self.dimgs])

        return self._jpegs


    @staticmethod
    def path2frame(path):
        try:
            expr = '(?<=_)[\d]+(?=[_.])'
            return int(re.findall(expr, os.path.basename(path))[-1])

        except ValueError:
            print 'Path has no frame number:', path

        return None

from PIL import Image


class jpegimageset():
    def __init__(self, jpegs):
        self.jpegs = jpegs

    @property
    def dtype(self):
        return np.uint8

    @property
    def max(self):
        return np.max(Image.open(self.jpegs[0]))

    @property
    def min(self):
        return np.min(Image.open(self.jpegs[0]))

    @property
    def ndim(self):
        return 3

    @property
    def shape(self):
        return (len(self.jpegs), np.shape(Image.open(self.jpegs[0]))[0], np.shape(Image.open(self.jpegs[0]))[1])

    @property
    def size(self):
        return len(self.jpegs) * np.product(np.size(Image.open(self.jpegs[0])))

    def __getitem__(self, item):
        print 'item:', item
        if type(item) in (int, np.int64):
            return self.jpegs[item]
        else:
            return np.array([self.jpegs[i] for i in item])


class StackImage(object):
    ndim = 3
    def __init__(self, filepath=None, data=None):
        # super(StackImage, self).__init__()
        self._rawdata = None
        self.filepath = filepath

        if filepath is not None:
            self.fabimage = fabio.open(filepath)
        elif data is not None:
            self.fabimage = data
        else:
            if filepath is None and data is None:
                raise ValueError('Either data or path to file must be provided')
        self.header = self.fabimage.header

        self._framecache = dict()
        self._cachesize = 2
        self.currentframe = 0

        raw = self.rawdata
        self.dtype = raw.dtype
        self.max = np.max(raw)
        self.min = np.min(raw)
        self.shape = len(self.fabimage),raw.shape[0],raw.shape[1]
        self.size = np.product(self.shape)

    @property
    def rawdata(self):
        # 'Permanently' cached
        if self._rawdata is None:
            self._rawdata = self._getframe()
        return self._rawdata


    def _getframe(self, frame=None): # keeps 3 frames in cache at most
        if frame is None: frame=self.currentframe
        if type(frame) is list and type(frame[0]) is slice:
            frame = 0 #frame[1].step
        self.currentframe = frame
        if frame not in self._framecache:
            # del the first cached item
            if len(self._framecache) > self._cachesize: del self._framecache[self._framecache.keys()[0]]
            self._framecache[frame] = self._getimage(frame)
        return self._framecache[frame]

    def _getimage(self, frame):
        return np.rot90(self.fabimage.getframe(frame).data, 3)

    def invalidatecache(self):
        self.cache = dict()
        print 'cache cleared'

    def __getitem__(self, item):
        return self._getframe(item)

    def __del__(self):
        try:
            self.fabimage.close()
        except ValueError:
            pass


class ProjectionStack(StackImage):
    def __init__(self, filepath=None, data=None):
        super(ProjectionStack, self).__init__(filepath=filepath, data=data)
        self.flats = self.fabimage.flats
        self.darks = self.fabimage.darks


class SinogramStack(StackImage):
    def __init__(self, filepath=None, data=None):
        super(SinogramStack, self).__init__(filepath=filepath, data=data)
        self._cachesize = 10

    def __new__(cls):
        cls.invalidatecache()

    @classmethod
    def cast(cls, obj):
        new_obj = copy(obj)
        new_obj.__class__ = cls
        new_obj.shape = new_obj.shape[2], new_obj.shape[0], new_obj.shape[1]
        return new_obj

    def _getimage(self, frame):
        return np.rot90(self.fabimage.getsinogram(frame).sinogram, 3)
