import numpy as np  # Import important packages
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import glob
from PIL import Image
import os
import re


app = QtGui.QApplication([])                   #Launches an app


def calcscale(imv):  # Defines calcscale function
    """

    """
    image = imv.getProcessedImage()

    scale = imv.scalemax / float(image[imv.currentIndex].shape[1])
    return scale


class imagetimeline(list):  # Sets up the image so it will fin the the viewer

    @property
    def shape(self):  # Defines shape function
        return (len(self), self[-1].shape[0], self[-1].shape[0])

    def __getitem__(self, item):  # Defines getitem function
        return list.__getitem__(self, item)

    @property
    def ndim(self):  # Defines ndim function
        return 3

    @property
    def size(self):  # Defines size functon
        return sum(map(np.size, self))


class TimelineView(pg.ImageView):  # Beginnings the class Timelineview
    def __init__(self, scalemax, *args, **kwargs):
        super(TimelineView, self).__init__(*args, **kwargs)
        self.scalemax = scalemax

    def quickMinMax(self, data):  # Defines quickMinMax functon
        return min(map(np.min, data)), max(map(np.max, data))

    def updateImage(self, autoHistogramRange=True):  # Defines updateImage functon
        if self.image is None:
            return

        scale = calcscale(self)  # Scales the image
        image = self.getProcessedImage()

        if autoHistogramRange:  # Sets the Y axis intensity bar
            self.ui.histogram.setHistogramRange(self.levelMin, self.levelMax)
        if self.axes['t'] is None:
            self.imageItem.updateImage(image)
        else:
            self.ui.roiPlot.show()
            self.imageItem.updateImage(image[self.currentIndex])

        self.imageItem.resetTransform()  # Resets the scale up below
        self.imageItem.scale(scale, scale)  # Scales up by the factor of scale
        print 'Image shape' + str(image.shape)
        print 'Scale set to: ' + str(scale)


tabwidget = QtGui.QTabWidget()

root = '/Users/austinblair/Downloads/test_20150714_144045/'

paths = glob.glob(os.path.join(root,
                               '*[0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9]_model.tif'))

indices = dict(zip(paths, [re.findall('\d{4}', os.path.basename(path)) for path in paths]))

tiles = dict()

for path, ind in indices.iteritems():
    if int(ind[1]) in tiles:
        tiles[int(ind[1])].append(path)
    else:
        tiles[int(ind[1])] = [path]

for tile, paths in tiles.iteritems():
    d = dict()
    im = Image.open(path).convert('L')
    imarray = np.array(im)

    print path  # Prints the path
    print imarray.shape  # Prints the shape of the array

    filename = os.path.basename(path)
    frame = os.path.splitext(filename)
    frame = frame[0]

    d[frame] = imarray

    data0 = imagetimeline(d.values())

    scalemax0 = max(map(np.shape, data0))[0]

    cw0 = QtGui.QWidget()

    tabwidget.addTab(cw0, u"tile" + str(tile))
    l0 = QtGui.QGridLayout()
    cw0.setLayout(l0)
    imv1_0 = TimelineView(scalemax0)
    l0.addWidget(imv1_0, 0, 0)
    imv1_0.setImage(data0)

    scale = calcscale(imv1_0)  # Sets up the scale
    imv1_0.imageItem.resetTransform()
    imv1_0.imageItem.scale(scale, scale)
    imv1_0.autoRange()






# d = dict()  # Create a directory for paths
# paths = glob.glob(
#     '/Users/austinblair/Downloads/test_20150714_144045/*[0-9][0-9][0-9][0-9]_0000_[0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9]_model.tif')
# for path in paths:
#     if "final" in path:
#         continue
#     im = Image.open(path).convert('L')
#     imarray = np.array(im)
#
#     print path  # Prints the path
#     print imarray.shape  # Prints the shape of the array
#
#     filename = os.path.basename(path)
#     frame = os.path.splitext(filename)
#     frame = frame[0]
#
#     d[frame] = imarray
#
# e = dict()  # Create a directory for paths
# paths = glob.glob(
#     '/Users/austinblair/Downloads/test_20150714_144045/*[0-9][0-9][0-9][0-9]_0001_[0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9]_model.tif')
# for path in paths:
#     if "final" in path:
#         continue
#     im = Image.open(path).convert('L')
#     imarray = np.array(im)
#
#     print path  # Prints the path
#     print imarray.shape  # Prints the shape of the array
#
#     filename = os.path.basename(path)
#     frame = os.path.splitext(filename)
#     frame = frame[0]
#
#     e[frame] = imarray
#
# f = dict()  # Create a directory for paths
# paths = glob.glob(
#     '/Users/austinblair/Downloads/test_20150714_144045/*[0-9][0-9][0-9][0-9]_0002_[0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9]_model.tif')
# for path in paths:
#     if "final" in path:
#         continue
#     im = Image.open(path).convert('L')
#     imarray = np.array(im)
#
#     print path  # Prints the path
#     print imarray.shape  # Prints the shape of the array
#
#     filename = os.path.basename(path)
#     frame = os.path.splitext(filename)
#     frame = frame[0]
#
#     f[frame] = imarray
#
# data0 = imagetimeline(d.values())
# data1 = imagetimeline(e.values())
# data2 = imagetimeline(f.values())
#
# scalemax0 = max(map(np.shape, data0))[0]
# scalemax1 = max(map(np.shape, data1))[0]
# scalemax2 = max(map(np.shape, data2))[0]
#
win = QtGui.QMainWindow()  # Create window with two ImageView widgets
win.resize(800, 800)
win.setWindowTitle('pyqtgraph example: Hiprmc ')
# cw = QtGui.QTabWidget()
#
#
# cw0 = QtGui.QWidget()
# cw1 = QtGui.QWidget()
# cw2 = QtGui.QWidget()
#
#
# tabwidget.addTab(cw0, u"tile0")
# tabwidget.addTab(cw1, u"tile1")
# tabwidget.addTab(cw2, u"tile2")
# tabwidget.setDocumentMode(True)
win.setCentralWidget(tabwidget)
# l0 = QtGui.QGridLayout()
# l1 = QtGui.QGridLayout()
# l2 = QtGui.QGridLayout() 
# cw0.setLayout(l0)
# cw1.setLayout(l1)
# cw2.setLayout(l2)
#
# imv1_0 = TimelineView(scalemax0)
# imv1_1 = TimelineView(scalemax1)
# imv1_2 = TimelineView(scalemax2)
#
# l0.addWidget(imv1_0, 0, 0)
# l1.addWidget(imv1_1, 0, 0)
# l2.addWidget(imv1_2, 0, 0)
#
win.show()
#
# imv1_0.setImage(data0)  # Display the data
# imv1_0.setHistogramRange(-0.01, 0.01)
# imv1_0.setLevels(-0.003, 0.003)
#
# imv1_1.setImage(data1)  # Display the data
# imv1_1.setHistogramRange(-0.01, 0.01)
# imv1_1.setLevels(-0.003, 0.003)
#
# imv1_2.setImage(data2)  # Display the data
# imv1_2.setHistogramRange(-0.01, 0.01)
# imv1_2.setLevels(-0.003, 0.003)
#
# scale = calcscale(imv1_0)  # Sets up the scale
# imv1_0.imageItem.resetTransform()
# imv1_0.imageItem.scale(scale, scale)
# imv1_0.autoRange()
#
# scale = calcscale(imv1_1)  # Sets up the scale
# imv1_1.imageItem.resetTransform()
# imv1_1.imageItem.scale(scale, scale)
# imv1_1.autoRange()
#
# scale = calcscale(imv1_2)  # Sets up the scale
# imv1_2.imageItem.resetTransform()
# imv1_2.imageItem.scale(scale, scale)
# imv1_2.autoRange()



if __name__ == '__main__':  # Start Qt event loop unless running in interactive mode.
    import sys

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
