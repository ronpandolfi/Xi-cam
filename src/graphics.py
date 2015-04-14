# --coding: utf-8 --
import pyqtgraph as pg
import numpy as np
from pyFAI import detectors
import scipy

import integration
import center_approx
from PySide.QtGui import QHBoxLayout

from PySide.QtGui import QWidget
from PySide.QtGui import QLabel
from PySide.QtCore import Qt
from PySide.QtGui import QAction
import cosmics
import fabio
import cv2
import visvis as vv
from PySide.QtGui import QStackedLayout


class imageTabTracker(QWidget):
    def __init__(self, paths, experiment, parent, operation=None):
        '''
        A collection of references that can be used to make an imageTab dynamically and dispose of it when unneeded
        :type path: str
        :param path:
        :param experiment:
        :param parent:
        :return:
        '''
        super(imageTabTracker, self).__init__()

        # When tab is activated, load an image tab and put it inside.

        #Whent tab is deactivated, dispose all of its objects but retain a reference to its constructor paramters

        self.paths = paths

        self.experiment = experiment
        self.parent = parent
        self.operation = operation
        self.tab = None


        parent.listmodel.widgetchanged()

        #self.load()
        self.isloaded = False


    def load(self):
        if not self.isloaded:
            if self.operation is None:
                imgdata = fabio.open(self.paths).data
                self.parent.ui.findChild(QLabel, 'filenamelabel').setText(self.paths)
            else:
                imgdata = [fabio.open(path).data for path in self.paths]

                imgdata = self.operation(imgdata)
                print(imgdata)

            self.layout = QHBoxLayout(self)
            self.tab = imageTab(imgdata, self.experiment, self.parent)
            self.layout.addWidget(self.tab)

            print('Successful load! :P')
            self.isloaded = True



    def unload(self):
        if self.isloaded:
            self.layout.parent = None
            self.layout.deleteLater()
            # self.tab = None
            print('Successful unload!')
            self.isloaded = False



class imageTab(QWidget):
    def __init__(self, imgdata, experiment, parent):
        '''
        A tab containing an imageview and plotview set in a splitter. Also manages functionality connected to a specific tab (masking/integration)
        :param imgdata:
        :param experiment:
        :return:
        '''
        super(imageTab, self).__init__()
        self.region = None
        self.layout = QStackedLayout(self)


        # Save image data and the experiment
        self.imgdata = np.rot90(imgdata, 2)
        self.experiment = experiment
        self.parentwindow = parent

        # Immediately mask any negative pixels #####MAKE THIS UNIQUE
        self.experiment.addtomask(self.imgdata < 0)

        # For storing what action is active (mask/circle fit...)
        self.activeaction = None

        # Make an imageview for the image
        # self.imgview = pg.ImageView(self)
        #self.imgview.setImage(self.imgdata.T)
        #self.imgview.autoRange()
        #self.imageitem = self.imgview.getImageItem()
        self.viewbox = pg.ViewBox(enableMenu=False)
        self.imageitem = pg.ImageItem()
        self.viewbox.addItem(self.imageitem)
        self.graphicslayoutwidget = pg.GraphicsLayoutWidget()
        self.graphicslayoutwidget.addItem(self.viewbox)
        self.viewbox.setAspectLocked(True)
        self.imghistLUT = pg.HistogramLUTItem(self.imageitem)
        self.graphicslayoutwidget.addItem(self.imghistLUT, 0, 1)

        # cross hair
        linepen = pg.mkPen('#FFA500')
        self.vLine = pg.InfiniteLine(angle=90, movable=False, pen=linepen)
        self.hLine = pg.InfiniteLine(angle=0, movable=False, pen=linepen)
        self.vLine.setVisible(False)
        self.hLine.setVisible(False)
        self.viewbox.addItem(self.vLine, ignoreBounds=True)
        self.viewbox.addItem(self.hLine, ignoreBounds=True)



        # Add a thin border to the image so it is visible on black background
        self.imageitem.border = pg.mkPen('w')

        self.coordslabel = QLabel('')
        self.layout.addWidget(self.coordslabel)
        self.coordslabel.setAlignment(Qt.AlignHCenter | Qt.AlignBottom)
        self.coordslabel.setStyleSheet("background-color: rgba(0,0,0,0%)")
        self.graphicslayoutwidget.scene().sigMouseMoved.connect(self.mouseMoved)
        self.layout.setStackingMode(QStackedLayout.StackAll)
        self.coordslabel.mouseMoveEvent = self.graphicslayoutwidget.mouseMoveEvent
        self.coordslabel.mousePressEvent = self.graphicslayoutwidget.mousePressEvent
        self.coordslabel.mouseReleaseEvent = self.graphicslayoutwidget.mouseReleaseEvent
        self.coordslabel.mouseDoubleClickEvent = self.graphicslayoutwidget.mouseDoubleClickEvent
        self.coordslabel.mouseGrabber = self.graphicslayoutwidget.mouseGrabber
        self.coordslabel.wheelEvent = self.graphicslayoutwidget.wheelEvent
        self.coordslabel.leaveEvent = self.graphicslayoutwidget.leaveEvent
        self.coordslabel.enterEvent = self.graphicslayoutwidget.enterEvent
        self.coordslabel.setMouseTracking(True)

        # Make a layout for the tab
        backwidget = QWidget()
        self.layout.addWidget(backwidget)
        self.backlayout = QHBoxLayout(backwidget)
        self.backlayout.setContentsMargins(0, 0, 0, 0)
        self.backlayout.addWidget(self.graphicslayoutwidget)

        # Add the Log Intensity check button to the context menu and wire up
        # self.imgview.buildMenu()
        #menu = self.viewbox.menu
        #self.actionLogIntensity = QAction('Log Intensity', menu, checkable=True)
        #self.actionLogIntensity.triggered.connect(self.logintensity)
        #menu.addAction(self.actionLogIntensity)
        #self.imgview.buildMenu()

        # Add a placeholder image item for the mask to the viewbox
        self.maskimage = pg.ImageItem(opacity=.25)
        self.viewbox.addItem(self.maskimage)

        ##
        # self.findcenter()
        # self.calibrate()

        if self.experiment.iscalibrated:
            self.replot()
            ##
            self.drawcenter()


        # self.maskoverlay()
        #self.updatelogintensity()
        self.redrawimage()

    def mouseMoved(self, evt):
        pos = evt  ## using signal proxy turns original arguments into a tuple
        if self.viewbox.sceneBoundingRect().contains(pos):
            mousePoint = self.viewbox.mapSceneToView(pos)
            index = int(mousePoint.x())
            if (0 < mousePoint.x() < self.imgdata.shape[1]) & (
                    0 < mousePoint.y() < self.imgdata.shape[0]):  # within bounds
                #angstrom=QChar(0x00B5)
                self.coordslabel.setText(u"<span style='font-size: 12pt;background-color:black;'>x=%0.1f,"
                                         u"   <span style=''>y=%0.1f</span>,   <span style=''>I=%0.1f</span>,"
                                         u"  q=%0.3f \u212B\u207B\u00B9,  q<sub>z</sub>=%0.3f \u212B\u207B\u00B9,"
                                         u"  q<sub>\u2225\u2225</sub>=%0.3f \u212B\u207B\u00B9</span>" % (
                                         mousePoint.x(),
                                         mousePoint.y(),
                                         self.imgdata[int(mousePoint.y()),
                                                      int(mousePoint.x())],
                                         pixel2q(mousePoint.x(),
                                                 mousePoint.y(),
                                                 self.experiment),
                                         pixel2q(None,
                                                 mousePoint.y(),
                                                 self.experiment),
                                         pixel2q(mousePoint.x(),
                                                 None,
                                                 self.experiment)))
                self.coordslabel.setVisible(True)
                self.coordslabel
            else:
                self.coordslabel.setVisible(False)
            self.vLine.setPos(mousePoint.x())
            self.hLine.setPos(mousePoint.y())

    def leaveEvent(self, evt):
        self.hLine.setVisible(False)
        self.vLine.setVisible(False)
        self.coordslabel.setVisible(False)

    def enterEvent(self, evt):
        self.hLine.setVisible(True)
        self.vLine.setVisible(True)


    def redrawimage(self):
        islogintensity = self.parentwindow.ui.findChild(QAction, 'actionLog_Intensity').isChecked()
        isradialsymmetry = self.parentwindow.ui.findChild(QAction, 'actionRadial_Symmetry').isChecked()
        ismirrorsymmetry = self.parentwindow.ui.findChild(QAction, 'actionMirror_Symmetry').isChecked()
        ismaskshown = self.parentwindow.ui.findChild(QAction, 'actionShow_Mask').isChecked()
        iscake = self.parentwindow.ui.findChild(QAction, 'actionCake').isChecked()
        img = self.imgdata.T.copy()

        # When the log intensity button toggles, switch the log scaling on the image
        if islogintensity:
            img = (np.log(img * (img > 0) + (img < 1)))
        imtest(img)
        if isradialsymmetry:
            centerx = self.experiment.getvalue('Center X')
            centery = self.experiment.getvalue('Center Y')
            symimg = np.rot90(img.copy(), 2)
            imtest(symimg)
            xshift = -(img.shape[0] - 2 * centerx)
            yshift = -(img.shape[1] - 2 * centery)
            symimg = np.roll(symimg, int(xshift), axis=0)
            symimg = np.roll(symimg, int(yshift), axis=1)
            imtest(symimg)
            marginmask = 1 - (detectors.ALL_DETECTORS[self.experiment.getvalue('Detector')]().calc_mask().T)
            imtest(marginmask)

            x, y = np.indices((img.shape))
            padmask = ((yshift < y) & (y < (yshift + img.shape[1])) & (xshift < x) & (x < (xshift + img.shape[0])))
            imtest(padmask)
            imtest(symimg * padmask * (1 - marginmask))
            img = img * (marginmask) + symimg * padmask * (1 - marginmask)

        elif ismirrorsymmetry:
            centery = self.experiment.getvalue('Center Y')
            symimg = np.fliplr(img.copy())
            imtest(symimg)
            yshift = -(img.shape[1] - 2 * centery)
            symimg = np.roll(symimg, int(yshift), axis=1)
            imtest(symimg)
            marginmask = 1 - (detectors.ALL_DETECTORS[self.experiment.getvalue('Detector')]().calc_mask().T)
            imtest(marginmask)

            x, y = np.indices((img.shape))
            padmask = ((yshift < y) & (y < (yshift + img.shape[1])))
            imtest(padmask)
            imtest(symimg * padmask * (1 - marginmask))
            img = img * (marginmask) + symimg * padmask * (1 - marginmask)

        if iscake:
            img, x, y = integration.cake(img, self.experiment)

        if ismaskshown:
            self.maskimage.setImage(np.dstack((
                self.experiment.mask.T, np.zeros_like(self.experiment.mask).T, np.zeros_like(self.experiment.mask).T,
                self.experiment.mask.T)), opacity=.25)
        else:
            self.maskimage.clear()

        self.imageitem.setImage(img)

    def linecut(self):
        if self.parentwindow.ui.findChild(QAction, 'actionLine_Cut').isChecked():
            self.region = pg.LineSegmentROI(
                [[self.experiment.getvalue('Center X'), self.experiment.getvalue('Center Y')],
                 [self.experiment.getvalue('Center X'), self.imgdata.shape[0]]])
            self.viewbox.addItem(self.region)
            self.replot()
            self.region.sigRegionChanged.connect(self.replot)
        else:
            self.viewbox.removeItem(self.region)
            self.region = None
            self.replot()

            # if self.parentwindow.ui.findChild(QAction,'actionVertical_Cut').isChecked():
            #    try:
            #        self.viewbox.removeItem(self.region)
            #    except AttributeError:
            #        print('Attribute error in verticalcut')
            #    self.region = pg.LinearRegionItem(orientation=pg.LinearRegionItem.Vertical,brush=pg.mkBrush('#00FFFF32'),bounds=[0,self.imgdata.shape[1]],values=[self.experiment.getvalue('Center X')-10,10+self.experiment.getvalue('Center X')])
            #    for line in self.region.lines:
            #        line.setPen(pg.mkPen('#00FFFF'))
            #    self.region.sigRegionChangeFinished.connect(self.replot)
            #    self.viewbox.addItem(self.region)
            #else:
            #    self.viewbox.removeItem(self.region)
            #    self.region = None

    # def horizontalcut(self):
    #    if self.parentwindow.ui.findChild(QAction,'actionHorizontal_Cut').isChecked():
    #        try:
    #            self.viewbox.removeItem(self.region)
    #        except AttributeError:
    #            print('Attribute error in horizontalcut')
    #        self.region = pg.LinearRegionItem(orientation=pg.LinearRegionItem.Horizontal,brush=pg.mkBrush('#00FFFF32'),bounds=[0,self.imgdata.shape[0]],values=[self.experiment.getvalue('Center Y')-10,10+self.experiment.getvalue('Center Y')])
    #        for line in self.region.lines:
    #            line.setPen(pg.mkPen('#00FFFF'))
    #        self.viewbox.addItem(self.region)
    #    else:
    #        self.viewbox.removeItem(self.region)
    #        self.region = None


    def removecosmics(self):
        c = cosmics.cosmicsimage(self.imgdata)
        c.run(maxiter=4)
        self.experiment.addtomask(c.mask)
        #self.maskoverlay()

    def findcenter(self):
        # Auto find the beam center
        [x, y] = center_approx.center_approx(self.imgdata)

        # Set the center in the experiment
        self.experiment.setValue('Center X', x)
        self.experiment.setValue('Center Y', y)
        self.drawcenter()

    def drawcenter(self):
        # Mark the center
        self.centerplot = pg.ScatterPlotItem([self.experiment.getvalue('Center X')],
                                             [self.experiment.getvalue('Center Y')], pen=None, symbol='o')
        self.viewbox.addItem(self.centerplot)

    def calibrate(self):
        # Choose detector
        self.detector = self.finddetector()

        self.findcenter()

        y, x = np.indices((self.imgdata.shape))
        r = np.sqrt((x - self.experiment.getvalue('Center X')) ** 2 + (y - self.experiment.getvalue('Center Y')) ** 2)
        r = r.astype(np.int)
        tbin = np.bincount(r.ravel(), self.imgdata.ravel())
        nr = np.bincount(r.ravel(), self.experiment.mask.ravel())
        with np.errstate(divide='ignore', invalid='ignore'):
            radialprofile = tbin / nr

        # Find peak positions, they represent the radii
        peaks = scipy.signal.find_peaks_cwt(np.nan_to_num(np.log(radialprofile + 3)), np.arange(30, 100))

        # Get the tallest peak
        bestpeak = peaks[radialprofile[peaks].argmax()]

        # Calculate sample to detector distance for lowest q peak
        tth = 2 * np.arcsin(0.5 * self.experiment.getvalue('Wavelength') / 58.367e-10)
        tantth = np.tan(tth)
        sdd = bestpeak * self.experiment.getvalue('Pixel Size X') / tantth

        self.experiment.setValue('Detector Distance', sdd)
        self.experiment.iscalibrated = True

        self.replot()
        # self.maskoverlay()

    def replot(self):
        self.parentwindow.integration.clear()

        if self.parentwindow.ui.findChild(QAction, 'actionMultiPlot').isChecked():
            self.replotothers()

        self.replotprimary()

    def replotprimary(self):
        cut = None

        if self.parentwindow.ui.findChild(QAction, 'actionLine_Cut').isChecked():
            # regionbounds=self.region.getRegion()
            #cut = np.zeros_like(self.imgdata)
            #cut[regionbounds[0]:regionbounds[1],:]=1
            cut = self.region.getArrayRegion(self.imgdata.T, self.imageitem)

            #self.q=pixel2q(np.arange(-self.imgdata.shape[0]/2,self.imgdata.shape[0]/2,1))
            x = np.linspace(self.viewbox.mapSceneToView(self.region.getSceneHandlePositions(0)[1]).x(),
                            self.viewbox.mapSceneToView(self.region.getSceneHandlePositions(1)[1]).x(),
                            cut.__len__())
            y = np.linspace(self.viewbox.mapSceneToView(self.region.getSceneHandlePositions(0)[1]).y(),
                            self.viewbox.mapSceneToView(self.region.getSceneHandlePositions(1)[1]).y(),
                            cut.__len__())
            #self.viewbox.mapToItem(self.imageitem,self.viewbox.mapToScene(self.region.getSceneHandlePositions(0)[1]))
            q = pixel2q(x, y, self.experiment)
            qmiddle = q.argmin()
            leftq = -q[0:qmiddle]
            rightq = q[qmiddle:]

            if leftq.__len__() > 1: self.parentwindow.integration.plot(leftq, cut[:qmiddle])
            if rightq.__len__() > 1: self.parentwindow.integration.plot(rightq, cut[qmiddle:])

        else:

            # Radial integraion
            self.q, self.radialprofile = integration.radialintegratepyFAI(self.imgdata, self.experiment,
                                                                          mask=self.experiment.mask)
            # Replot
            self.parentwindow.integration.plot(self.q, self.radialprofile)

    def replotothers(self):
        for tab in self.parentwindow.ui.findChildren(imageTab):
            tab.replotassecondary(self.parentwindow.integration)

    def replotassecondary(self, plotitem):
        plotitem.plot(self.q, self.radialprofile, pen=pg.mkPen(0.5))


    def polymask(self):
        if self.activeaction is None:  # If there is no active action
            self.activeaction = 'polymask'

            # Start with a box around the center
            left = self.experiment.getvalue('Center X') - 100
            right = self.experiment.getvalue('Center X') + 100
            up = self.experiment.getvalue('Center Y') - 100
            down = self.experiment.getvalue('Center Y') + 100

            # Add ROI item to the image
            self.maskROI = pg.PolyLineROI([[left, up], [left, down], [right, down], [right, up]], pen=(6, 9),
                                          closed=True)
            self.viewbox.addItem(self.maskROI)

            # Override the ROI's function to check if any points will be moved outside the boundary; False prevents move
            def checkPointMove(handle, pos, modifiers):
                p = self.viewbox.mapToView(pos)
                if 0 < p.y() < self.imgdata.shape[0] and 0 < p.x() < self.imgdata.shape[1]:
                    return True
                else:
                    return False

            self.maskROI.checkPointMove = checkPointMove

        elif self.activeaction == 'polymask':  # If the mask is completed
            self.activeaction = None

            # Get the region of the image that was selected; unforunately the region is trimmed
            maskedarea = self.maskROI.getArrayRegion(np.ones_like(self.imgdata.T), self.imageitem,
                                                     returnMappedCoords=True)  # levels=(0, arr.max()
            print maskedarea.shape

            # Decide how much to left and top pad based on the ROI bounding rectangle
            boundrect = self.viewbox.itemBoundingRect(self.maskROI)
            leftpad = boundrect.x()
            toppad = boundrect.y()

            # Pad the mask so it has the same shape as the image
            maskedarea = np.pad(maskedarea, ((int(leftpad), 0), (int(toppad), 0)), mode='constant').T
            maskedarea = np.pad(maskedarea, (
                (0, self.imgdata.shape[0] - maskedarea.shape[0]), (0, self.imgdata.shape[1] - maskedarea.shape[1])),
                                mode='constant')

            # Add the masked area to the active mask
            self.experiment.addtomask(maskedarea)

            # Draw the overlay
            #self.maskoverlay()

            # Remove the ROI
            self.viewbox.removeItem(self.maskROI)

            # Redo the integration
            self.replot()

    def maskoverlay(self):
        # Draw the mask as a red channel image with an alpha mask
        self.maskimage.setImage(np.dstack((
            self.experiment.mask.T, np.zeros_like(self.experiment.mask).T, np.zeros_like(self.experiment.mask).T,
            self.experiment.mask.T)), opacity=.25)


    def finddetector(self):
        for name, detector in detectors.ALL_DETECTORS.iteritems():
            if detector.MAX_SHAPE == self.imgdata.shape:
                detector = detector()
                mask = detector.calc_mask()
                self.experiment.addtomask(mask)
                self.experiment.setValue('Pixel Size X', detector.pixel1)
                self.experiment.setValue('Pixel Size Y', detector.pixel2)
                self.experiment.setValue('Detector', name)
                return detector


class smallimageview(pg.GraphicsLayoutWidget):
    def __init__(self, model):
        super(smallimageview, self).__init__()
        self.model = model
        # self.setLayout(QHBoxLayout())
        #self.layout().setContentsMargins(0, 0, 0, 0)
        self.view = self.addViewBox(lockAspect=True)

        self.imageitem = pg.ImageItem()
        self.view.addItem(self.imageitem)
        # self.setMaximumHeight(100)
        # self.addItem(self.imageitem)

    def loaditem(self, index):
        path = self.model.filePath(index)
        self.imgdata = fabio.open(path).data
        self.imageitem.setImage(np.rot90(np.log(self.imgdata * (self.imgdata > 0) + (self.imgdata < 1)), 3),
                                autoLevels=True)


def imtest(image):
    if False:
        image = image * 255 / image.max()
        cv2.imshow('step?', cv2.resize(image.astype(np.uint8), (0, 0), fx=.2, fy=.2))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def pixel2q(x, y, experiment):
    if x is None:
        x = experiment.getvalue('Center X')
    if y is None:
        y = experiment.getvalue('Center Y')

    r = np.sqrt((x - experiment.getvalue('Center X')) ** 2 + (y - experiment.getvalue('Center Y')) ** 2)
    theta = np.arctan2(r * experiment.getvalue('Pixel Size X'),
                       experiment.getvalue('Detector Distance'))
    # theta=x*self.config.getfloat('Detector','Pixel Size')*0.000001/self.config.getfloat('Beamline','Detector Distance')
    wavelength = experiment.getvalue('Wavelength')
    return 4 * np.pi / wavelength * np.sin(theta / 2) * 1e-10