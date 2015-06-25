import fabio
import numpy
import pyfits
import os
import numpy as np
from nexpy.api import nexus as nx
from pipeline import detectors
import glob
import re
import time
import scipy.ndimage
import writer

acceptableexts = '.fits .edf .tif .nxs .tif'


def loadsingle(path):
    return loadimage(path), loadparas(path)


def loadimage(path):
    data = None
    try:
        if os.path.splitext(path)[1] in acceptableexts:
            if os.path.splitext(path)[1] == '.gb':
                raise NotImplementedError('Format not yet implemented.')  # data = numpy.loadtxt()
            elif os.path.splitext(path)[1] == '.fits':
                data = pyfits.open(path)[2].data
                return data
            elif os.path.splitext(path)[1] == '.nxs':
                nxroot = nx.load(path)
                # print nxroot.tree
                if hasattr(nxroot.data, 'signal'):
                    data = nxroot.data.signal
                    return data
                else:
                    return loadsingle(str(nxroot.data.rawfile))

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
                print head[0].header.keys()
                paras = scanparaslines(str(head[0].header).split('\r'))
                print paras
            elif os.path.splitext(path)[1] == '.nxs':
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
            print 'Attempt', i + 1, 'to read', path, 'succeded; continuing...'
            return int(nxroot.data.variation)
        except IOError:
            print 'Could not load', path, ', trying again in 0.2 s'
            time.sleep(0.2)
            nxroot = nx.load(path)

    return None



def loadparas(path):
    try:
        extension = os.path.splitext(path)[1]
        if extension == '.fits':
            head = pyfits.open(path)
            # print head[0].header
            return head[0].header

        elif extension == '.edf':
            txtpath = os.path.splitext(path)[0] + '.txt'
            if os.path.isfile(txtpath):
                return scanparas(txtpath)
            else:
                basename = os.path.splitext(path)[0]
                obj = re.search('_\d+$', basename)
                if obj is not None:
                    iend = obj.start()
                    token = basename[:iend] + '*txt'
                    txtpath = glob.glob(token)[0]
                    return scanparas(txtpath)
                else:
                    return None

        elif extension == '.gb':
            raise NotImplementedError('This format is not yet supported.')

        elif extension == '.nxs':
            nxroot = nx.load(path)
            # print nxroot.tree
            return nxroot

    except IOError:
        print('Unexpected read error in loadparas')
    except IndexError:
        print('No txt file found in loadparas')
    return None


def scanparas(path):
    with open(path, 'r') as f:
        lines = f.readlines()

    paras = scanparaslines(lines)

    return paras

def scanparaslines(lines):
    paras = dict()
    for line in lines:
        cells = filter(None, re.split('[=:]+', line))

        key = cells[0]

        if cells.__len__() == 2:
            cells[1] = cells[1].split('/')[0]
            paras[key] = cells[1]
        elif cells.__len__() == 1:
            paras[key] = cells[0]

    return paras


def loadstichted(filepath2, filepath1):
    (data1, paras1) = loadsingle(filepath1)
    (data2, paras2) = loadsingle(filepath2)

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
        padtop2 = abs(deltaY)
        padbottom1 = abs(deltaY)
    else:
        padtop1 = abs(deltaY)
        padbottom2 = abs(deltaY)

    if deltaX < 0:
        padleft2 = abs(deltaX)
        padright1 = abs(deltaX)

    else:
        padleft1 = abs(deltaX)
        padright2 = abs(deltaX)

    d2 = numpy.pad(data2, ((padtop2, padbottom2), (padleft2, padright2)), 'constant')
    d1 = numpy.pad(data1, ((padtop1, padbottom1), (padleft1, padright1)), 'constant')

    # mask2 = numpy.pad((data2 > 0), ((padtop2, padbottom2), (padleft2, padright2)), 'constant')
    #mask1 = numpy.pad((data1 > 0), ((padtop1, padbottom1), (padleft1, padright1)), 'constant')
    mask2 = numpy.pad(1 - (finddetector(data2.T)[1]), ((padtop2, padbottom2), (padleft2, padright2)), 'constant')
    mask1 = numpy.pad(1 - (finddetector(data1.T)[1]), ((padtop1, padbottom1), (padleft1, padright1)), 'constant')

    with numpy.errstate(divide='ignore'):
        data = (d1 + d2) / (mask2 + mask1)
    return data


def finddetector(imgdata):
    for name, detector in detectors.ALL_DETECTORS.iteritems():
        if hasattr(detector, 'MAX_SHAPE'):
            #print name, detector.MAX_SHAPE, imgdata.shape[::-1]
            if detector.MAX_SHAPE == imgdata.shape[::-1]:  #
                detector = detector()
                mask = detector.calc_mask()
                return name, mask, detector
        if hasattr(detector, 'BINNED_PIXEL_SIZE'):
            #print detector.BINNED_PIXEL_SIZE.keys()
            if imgdata.shape[::-1] in [tuple(np.array(detector.MAX_SHAPE) / b) for b in
                                       detector.BINNED_PIXEL_SIZE.keys()]:
                detector = detector()
                mask = detector.calc_mask()
                return name, mask, detector
    return None, None

def finddetectorbyfilename(path):
    imgdata = loadsingle(path)[0].T
    return finddetector(imgdata)

def loadthumbnail(path):
    nxpath = os.path.splitext(path)[0] + '.nxs'
    if os.path.isfile(nxpath):
        print nx.load(path).tree
        img = np.asarray(nx.load(path).data.thumbnail)
    else:
        img = loadimage(path)

    if img is not None:
        img = writer.thumbnail(img)
    return img


def loadpath(path):
    if '_lo_' in path:
        # print "input: lo / output: hi"
        path2 = path.replace('_lo_', '_hi_')
        return loadstichted(path, path2), None

    elif '_hi_' in path:
        # print "input: hi / output: lo"
        path2 = path.replace('_hi_', '_lo_')
        return loadstichted(path, path2), None

    else:
        # print "file don't contain hi or lo, just 1 file"
        return loadsingle(path)


    #
    # if __name__=='__main__':
    # from PIL import Image
    # data,paras=loadpath('/home/remi/PycharmProjects/saxs-on/samples/AgB_5s_hi_2m.edf')
    # print data
    # IM=Image.fromarray(data)
    #     IM.show()









    ##################################################################"
    #
    # tailleIM=(taille[0]-2*deltaY,taille[1])
    # IM=zeros((tailleIM))
    #
    # for i in range(0,np.size(IM,0)):
    #     for j in range(0,np.size(IM,1)):
    #         if data1[i,j]==0:
    #             IM[i,j]=data2[i+deltaY,j]
    #         elif data2[i+deltaY,j]==0:
    #             IM[i,j]=data1[i,j]
    #         else:
    #             IM[i,j]=(data1[i,j]+data2[i+deltaY,j])/2
    # Ima=Image.fromarray(IM)

# Ima.show()


import integration, remesh
from fabio import file_series


class diffimage():
    def __init__(self, filepath=None, data=None, detector=None, experiment=None):
        """
        Image class for diffraction images that caches and validates cache
        :param filepath: str
        :param data: numpy.multiarray.ndarray
        :param detector: pyFAI.detectors.Detector
        :param experiment: hipies.config.experiment
        """
        if data is not None:
            self.data = data
        elif filepath is not None:
            self.data = loadpath(filepath)

        self.filepath = filepath
        self._detector = detector
        self._params = None
        self.experiment = experiment


        ### All object I want to save that depend on config parameters must be cached in here instead!!!!
        self.cache = dict()
        self.cachecheck = None

    def checkcache(self):
        pass
        # compare experiment with cachecheck

    def invalidatecache(self):
        self.cache = dict()

    @property
    def params(self):
        if self._params is None:
            self._params = loadparas(self.paths[0])
        return self._params

    @property
    def detector(self):
        if self.detector is None:
            if self.filepath is not None:
                name, mask, detector = finddetectorbyfilename(self.filepath)
            elif self.data is not None:
                name, mask, detector = finddetector(self.data)
            else:
                return None

            self.detectorname = name
            self.mask = mask
            self._detector = detector
        return self._detector

    @detector.setter
    def detector(self, value):
        if type(value) == str:
            try:
                self._detector = detectors.ALL_DETECTORS[value]
            except KeyError:
                try:
                    self._detector = getattr(detectors, value)
                except AttributeError:
                    raise KeyError('Detector not found in pyFAI registry: ' + value)
                    return None
        else:
            self._detector = value

    @property
    def params(self):
        if self._params is None:
            self._params = loadparas(self.filepath)
        return self._params

    @property
    def thumb(self):
        if self._thumb is None:
            self._thumb = writer.thumbnail(self.data)
        return self._thumb

    def iscached(self, key):
        return hasattr(self.cache, key)

    @property
    def cake(self):
        if not self.iscached('cake'):
            cake, x, y = integration.cake(self.data, self.experiment)
            cakemask, _, _ = integration.cake(self.mask, self.experiment)
            cakemask = (cakemask > 0) * 255

            self.cache['cake'] = cake
            self.cache['cakemask'] = cakemask
            self.cache['cakeqx'] = x
            self.cache['cakeqy'] = y

        return self.cache['cake'], self.cache['cakemask']

    @property
    def remesh(self):
        if not self.iscached('remesh'):
            remeshdata, x, y = remesh.remesh(np.rot90(self.data, 1).copy(), self.filepath,
                                             self.experiment.getGeometry())
            remeshmask, _, _ = remesh.remesh(np.rot90(self.mask, 1).copy(), self.filepath,
                                             self.experiment.getGeometry())

            self.cache['remesh'] = remeshdata
            self.cache['remeshmask'] = remeshmask
            self.cache['remeshqx'] = x
            self.cache['remeshqy'] = y

        return self.cache['remesh'], self.cache['remeshmask']

    def __del__(self):
        self.writenexus()

    def writenexus(self):
        nxpath = os.path.splitext(self.filepath)[0] + '.nxs'
        writer.writenexus(self.data, self.thumb, nxpath, rawpath=self.filepath)


class imageseries(fabio.file_series.file_series):
    def __init__(self, paths, operation=None):
        super(imageseries, self).__init__(paths)
        self.operation = operation

    def currentdiffimage(self):
