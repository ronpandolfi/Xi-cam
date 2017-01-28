import numpy as np  # Import important packages
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import glob
from PIL import Image
from collections import OrderedDict
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

    sigImageChanged = QtCore.Signal()

    def __init__(self, scalemax, *args, **kwargs):
        super(TimelineView, self).__init__(*args, **kwargs)
        self.scalemax = scalemax

        self.view_label = QtGui.QLabel(self)
        self.view_label.setText('No: ')
        self.view_number = QtGui.QSpinBox(self)
        self.view_number.setReadOnly(True)
        self.view_number.setMaximum(10000)
        self.ui.gridLayout.addWidget(self.view_label, 1, 1, 1, 1)
        self.ui.gridLayout.addWidget(self.view_number, 1, 2, 1, 1)

        self.label = QtGui.QLabel(parent=self)
        self.ui.gridLayout.addWidget(self.label, 1, 0, 1, 1)



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

        self.view_number.setValue(self.currentIndex)
        self.sigImageChanged.emit()




class fftView(QtGui.QTabWidget):
    def __init__(self, *args, **kwargs):
        super(fftView, self).__init__(*args, **kwargs)
        self.img_dict = OrderedDict()

    def add_images(self, image_list, do_fft=True, loadingfactors=None):

        self.clear()

        if not image_list:
            return
        if type(image_list) == dict:
            for path, img in image_list.iteritems():
                try:
                    img = np.array(img)
                    len(img)
                    flag = True
                    if do_fft: img = self.do_fft(img)

                    for item in self.img_dict.itervalues():
                        if np.array_equal(img, item): flag = False
                    if flag:
                        self.img_dict[path.split('/')[-1]] = img
                except TypeError:
                    continue
        else:
            for img in image_list:
                try:
                    img = np.array(img)
                    len(img)
                    flag = True
                    if do_fft: img = self.do_fft(img)

                    for item in self.img_dict.itervalues():
                        if np.array_equal(img, item): flag = False
                    if flag:
                        self.img_dict[len(self.img_dict)] = img
                except TypeError:
                    continue

        # sort img dictionary by key
        img_dict = OrderedDict()
        for key in sorted(self.img_dict.keys()):
            img_dict[key] = self.img_dict[key]
        del self.img_dict
        self.img_dict = img_dict

        data = imagetimeline(self.img_dict.itervalues())
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
        view.sigImageChanged.connect(self.imageNameChanged)

        if loadingfactors is None:
            self.addTab(view, u"Tile " + str(1))
        else:
            self.addTab(view, str(loadingfactors))
        self.tabBar().hide()

        view.sigImageChanged.emit()

    def do_fft(self, img):
        """
        Returns absolute value of 2-D FFT of image. Assumes 2-D image
        """

        return np.log(np.fft.fftshift(np.abs(np.fft.fft2(img)**2)) + 1e-10)


    def open_from_rmcView(self, image_list):
        images = {}
        for lst in image_list:
            path = "/"
            for item in lst:
                path = os.path.join(path, item)
            img = Image.open(path).convert('L')
            img = np.array(img)
            images[path] = img

        self.add_images(images)

    def imageNameChanged(self):
        view = self.currentWidget()
        for key, val in self.img_dict.iteritems():
            if np.array_equal(val, view.image[view.currentIndex]):
                self.currentWidget().label.setText(key)


class rmcView(QtGui.QTabWidget):

    def __init__(self, root, loadingfactors=[None]):
        super(rmcView, self).__init__()

        self.image_list = []
        self.image_dict = OrderedDict()

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
                self.image_dict[path.split('/')[-1]] = img

            data = imagetimeline(images)

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
            view.sigImageChanged.connect(self.imageNameChanged)

            if loadingfactors is None:
                self.addTab(view, u"Tile " + str(tile + 1))
            else:
                self.addTab(view, str(loadingfactor))

            view.sigImageChanged.emit()




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
                self.image_dict[path.split('/')[-1]] = img

            data = imagetimeline(images)
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
            view.sigImageChanged.connect(self.imageNameChanged)


            if loadingfactors is None:
                self.addTab(view, u"Tile " + str(tile + 1))
            else:
                self.addTab(view, str(loadingfactor))

            view.sigImageChanged.emit()


    def imageNameChanged(self):

        view = self.currentWidget()
        for key, val in self.image_dict.iteritems():
            if np.array_equal(val, view.image[view.currentIndex]):
                self.currentWidget().label.setText(key)




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
