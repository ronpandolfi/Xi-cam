import numpy as np  # Import important packages
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import glob
from PIL import Image
import os
import re


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
    def ndim(self):  # Defines ndim functionq
        return 3

    @property
    def size(self):  # Defines size functon
        return sum(map(np.size, self))

    @property
    def max(self):
        return max(map(np.max, self))

    @property
    def min(self):
        return min(map(np.min, self))

    @property
    def dtype(self):
        return type(self[0][0, 0])


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

class fftView(QtGui.QTabWidget):
    def __init__(self, *args, **kwargs):
        super(fftView, self).__init__(*args, **kwargs)
        self.img_list = []

    def add_images(self, image_list, loadingfactors=None):

        self.clear()

        if not image_list:
            return

        for img in image_list:
            try:
                img = np.array(img)
                len(img)
                flag = True
                for item in self.img_list:
                    if np.array_equal(img, item): flag = False
                if flag:
                    self.img_list.append(img)
            except TypeError:
                continue

        data = imagetimeline(self.img_list)
        sizemax = max(map(np.shape, data))[0]

        view = TimelineView(sizemax)
        view.setImage(data)

        scale = calcscale(view)  # Sets up the scale
        view.imageItem.resetTransform()
        view.imageItem.scale(scale, scale)
        view.autoRange()
        view.getHistogramWidget().setHidden(False)
        view.ui.roiBtn.setHidden(True)
        view.ui.menuBtn.setHidden(True)

        if loadingfactors is None:
            self.addTab(view, u"Tile " + str(1))
        else:
            self.addTab(view, str(loadingfactors))
        self.tabBar().hide()


    def open_from_rmcView(self, image_list):
        images = []
        for lst in image_list:
            path = "/"
            for item in lst:
                path = os.path.join(path, item)
            img = Image.open(path).convert('L')
            img = np.array(img)

            images.append(img)
        self.add_images(images)

class rmcView(QtGui.QTabWidget):
    def __init__(self, root, loadingfactors=[None]):
        super(rmcView, self).__init__()

        self.image_list = []

        paths = glob.glob(os.path.join(root,
                                       '[0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9]_model.tif'))

        indices = dict(zip(paths, [re.findall('\d{4}', os.path.basename(path)) for path in paths]))

        tiles = dict()

        for path, ind in indices.iteritems():
            if int(ind[1]) in tiles:
                tiles[int(ind[1])].append(path)
            else:
                tiles[int(ind[1])] = [path]
            self.image_list.append(path.split('/'))

        for tile, loadingfactor in zip(tiles, loadingfactors):
            images = []
            paths = sorted(tiles[tile])
            for path in paths:
                img = Image.open(path).convert('L')
                img = np.array(img)

                print path  # Prints the path
                print img.shape  # Prints the shape of the array

                images.append(img)

            data = imagetimeline(images)

            sizemax = max(map(np.shape, data))[0]

            view = TimelineView(sizemax)
            view.setImage(data)

            scale = calcscale(view)  # Sets up the scale
            view.imageItem.resetTransform()
            view.imageItem.scale(scale, scale)
            view.autoRange()
            view.getHistogramWidget().setHidden(True)
            view.ui.roiBtn.setHidden(True)
            view.ui.menuBtn.setHidden(True)
            if loadingfactors is None:
                self.addTab(view, u"Tile " + str(tile + 1))
            else:
                self.addTab(view, str(loadingfactor))

    def addNewImages(self, root, loadingfactors=[None]):

        self.clear()

        paths = glob.glob(os.path.join(root,
                                       '[0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9]_model.tif'))

        indices = dict(zip(paths, [re.findall('\d{4}', os.path.basename(path)) for path in paths]))

        tiles = dict()


        for path, ind in indices.iteritems():
            if int(ind[1]) in tiles:
                tiles[int(ind[1])].append(path)
            else:
                tiles[int(ind[1])] = [path]

            if path.split('/') not in self.image_list:
                self.image_list.append(path.split('/'))

        for tile, loadingfactor in zip(tiles, loadingfactors):
            images = []
            paths = sorted(tiles[tile])
            for path in paths:
                img = Image.open(path).convert('L')
                img = np.array(img)

                print path  # Prints the path
                print img.shape  # Prints the shape of the array

                images.append(img)

            data = imagetimeline(images)
            sizemax = max(map(np.shape, data))[0]

            view = TimelineView(sizemax)
            view.setImage(data)

            scale = calcscale(view)  # Sets up the scale
            view.imageItem.resetTransform()
            view.imageItem.scale(scale, scale)
            view.autoRange()
            view.getHistogramWidget().setHidden(True)
            view.ui.roiBtn.setHidden(True)
            view.ui.menuBtn.setHidden(True)
            if loadingfactors is None:
                self.addTab(view, u"Tile " + str(tile + 1))
            else:
                self.addTab(view, str(loadingfactor))



if __name__ == '__main__':  # Start Qt event loop unless running in interactive mode.
    import sys

    app = QtGui.QApplication([])  # Launches an app
#    root = '/Users/austinblair/Downloads/test_20150714_144045/'
    root = '/home/hparks/Desktop/processed_20161201_170824'

#    win = QtGui.QMainWindow()  # Create window with two ImageView widgets
#    win.resize(800, 800)

    win = QtGui.QStackedWidget()
    win.setWindowTitle('pyqtgraph example: Hiprmc ')

#    win.setCentralWidget(rmcView(root))
    win.addWidget(rmcView(root))

    win.show()

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
