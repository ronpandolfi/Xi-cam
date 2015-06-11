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

acceptableexts = '.fits .edf .tif .nxs'


def loadsingle(path):
    """
    :type path : str
    :param path:
    :return:
    """
    # print os.path.splitext(path)
    try:
        if os.path.splitext(path)[1] in acceptableexts:
            if os.path.splitext(path)[1] == '.gb':
                data = numpy.loadtxt()
            if os.path.splitext(path)[1] == '.fits':
                data = pyfits.open(path)[2].data
                paras = loadparas(path)
                return data, paras
            elif os.path.splitext(path)[1] == '.nxs':
                nxroot = nx.load(path)
                # print nxroot.tree
                if hasattr(nxroot.data, 'signal'):
                    data = nxroot.data.signal
                    return data, nxroot
                else:
                    print ('here:', nxroot.data.rawfile)
                    return loadsingle(str(nxroot.data.rawfile))
                # print('here',data)

            else:
                data = fabio.open(path).data
                paras = loadparas(path)
                return data, paras
    except IOError:
        print('IO Error loading: ' + path)

    return None, None

    # except TypeError:
    #   print('Failed to load',path,', its probably not an image format I understand.')
    #  return None,None


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

def loadjustimage(path):
    data = fabio.open(path).data
    return data

def loadparas(path):
    try:
        if os.path.splitext(path)[1] == '.fits':
            head = pyfits.open(path)
            print head[0].header
            return head[0].header
        elif os.path.splitext(path)[1] == '.edf':

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
    except IOError:
        print('Unexpected read error in loadparas')
        # except IndexError:
        #print('No txt file found in loadparas')
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
            print name, detector.MAX_SHAPE, imgdata.shape[::-1]
            if detector.MAX_SHAPE == imgdata.shape[::-1]:  #
                detector = detector()
                mask = detector.calc_mask()
                return detector, mask
        if hasattr(detector, 'BINNED_PIXEL_SIZE'):
            print detector.BINNED_PIXEL_SIZE.keys()
            if imgdata.shape[::-1] in [tuple(np.array(detector.MAX_SHAPE) / b) for b in
                                       detector.BINNED_PIXEL_SIZE.keys()]:
                detector = detector()
                mask = detector.calc_mask()
                return detector, mask


def finddetectorbyfilename(path):
    imgdata = loadsingle(path)[0].T
    for name, detector in detectors.ALL_DETECTORS.iteritems():
        if hasattr(detector, 'MAX_SHAPE'):
            print name, detector.MAX_SHAPE, imgdata.shape[::-1]
            if detector.MAX_SHAPE == imgdata.shape[::-1]:  #
                detector = detector()
                mask = detector.calc_mask()
                return detector, mask
        if hasattr(detector, 'BINNED_PIXEL_SIZE'):
            print detector.BINNED_PIXEL_SIZE.keys()
            if imgdata.shape[::-1] in [tuple(np.array(detector.MAX_SHAPE) / b) for b in
                                       detector.BINNED_PIXEL_SIZE.keys()]:
                detector = detector()
                mask = detector.calc_mask()
                return detector, mask

def loadthumbnail(path):
    nxpath = os.path.splitext(path)[0] + '.nxs'
    if os.path.isfile(nxpath):
        print nx.load(path).tree
        img = np.asarray(nx.load(path).data.thumbnail)
    else:
        img, _ = loadsingle(path)

    if img is not None:
        img = np.log(img * (img > 0) + 1.)
        img *= 255 / np.max(np.asarray(img))

        img = img.astype(np.uint8)
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




