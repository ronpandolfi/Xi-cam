# --coding: utf-8 --

import os

from PySide import QtGui
from PySide import QtCore
from PySide.QtCore import Qt
import pyqtgraph as pg
import numpy as np
from pyFAI import detectors
from fabio import edfimage
import cv2

import integration
import center_approx
import cosmics
import loader
import peakfinding
import remesh


class imageTabTracker(QtGui.QWidget):
    def __init__(self, paths, experiment, parent, operation=None):
        """
        A collection of references that can be used to make an imageTab dynamically and dispose of it when unneeded
        :type paths: list[str]
        :type experiment: config.experiment
        :type parent: main.MyMainWindow
        :type operation:
        :return:
        """
        super(imageTabTracker, self).__init__()

        self.paths = paths
        self.experiment = experiment
        self.parent = parent
        self.operation = operation
        self.tab = None
        self.layout = None


        parent.listmodel.widgetchanged()

        #self.load()
        self.isloaded = False


    def load(self):
        """
        load this tab; rebuild the viewer
        """
        if not self.isloaded:
            if self.operation is None:

                try:
                    imgdata, paras = loader.loadpath(self.paths)
                except IOError:
                    print('File moved or deleted. Load failed')

                    return None
                self.parent.ui.findChild(QtGui.QLabel, 'filenamelabel').setText(self.paths)
            else:
                imgdata, paras = [loader.loadpath(path) for path in self.paths]

                imgdata = self.operation(imgdata)
                # print(imgdata)

            self.layout = QtGui.QHBoxLayout(self)
            self.layout.setContentsMargins(0, 0, 0, 0)
            self.tab = imageTab(imgdata, self.experiment, self.parent, self.paths)
            self.layout.addWidget(self.tab)
            self.isloaded = True
            self.tab.cache1Dintegration.connect(self.cache1Dintegration)
            self.tab.send1Dintegration()

    def cache1Dintegration(self, q, I):
        self.q = q
        self.I = I
        print('tab cached')


    def unload(self):
        """
        orphan the tab widgets and queue them for deletion. Mwahahaha.
        """
        if self.isloaded:
            for child in self.children():
                #print child
                if type(child) is imageTab:
                    self.layout.removeWidget(child)
                    child.deleteLater()
                if type(child) is QtGui.QHBoxLayout:
                    child.parent = None
                    child.deleteLater()
            self.tab = None
            self.layout = None
            #print('Successful unload!')
            self.isloaded = False


    def replotassecondary(self):

        self.parent.integration.plot(self.q, self.I, pen=pg.mkPen('#555555'))


class imageTab(QtGui.QWidget):
    cache1Dintegration = QtCore.Signal(np.ndarray, np.ndarray)

    def __init__(self, imgdata, experiment, parent, paths=None):
        """
        A tab containing an imageview. Also manages functionality connected to a specific tab (masking/integration)
        :param imgdata:
        :param experiment:
        :return:
        """
        super(imageTab, self).__init__()
        self.region = None
        self.maskROI = None
        self.layout = QtGui.QStackedLayout(self)
        self.path = paths

        # Save image data and the experiment
        self.imgdata = np.rot90(imgdata, 3)
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
        self.viewbox = pg.ViewBox()  # enableMenu=False)
        self.imageitem = pg.ImageItem()
        self.viewbox.addItem(self.imageitem)
        self.graphicslayoutwidget = pg.GraphicsLayoutWidget()
        self.graphicslayoutwidget.addItem(self.viewbox)
        self.viewbox.setAspectLocked(True)
        self.imghistLUT = pg.HistogramLUTItem(self.imageitem)
        self.graphicslayoutwidget.addItem(self.imghistLUT, 0, 1)
        self.imghistLUT.autoHistogramRange()



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

        self.coordslabel = QtGui.QLabel('')
        self.layout.addWidget(self.coordslabel)
        self.coordslabel.setAlignment(Qt.AlignHCenter | Qt.AlignBottom)
        self.coordslabel.setStyleSheet("background-color: rgba(0,0,0,0%)")
        self.graphicslayoutwidget.scene().sigMouseMoved.connect(self.mouseMoved)
        self.layout.setStackingMode(QtGui.QStackedLayout.StackAll)
        self.coordslabel.mouseMoveEvent = self.graphicslayoutwidget.mouseMoveEvent
        self.coordslabel.mousePressEvent = self.graphicslayoutwidget.mousePressEvent
        self.coordslabel.mouseReleaseEvent = self.graphicslayoutwidget.mouseReleaseEvent
        self.coordslabel.mouseDoubleClickEvent = self.graphicslayoutwidget.mouseDoubleClickEvent
        self.coordslabel.mouseGrabber = self.graphicslayoutwidget.mouseGrabber
        self.coordslabel.wheelEvent = self.graphicslayoutwidget.wheelEvent
        self.coordslabel.leaveEvent = self.graphicslayoutwidget.leaveEvent
        self.coordslabel.enterEvent = self.graphicslayoutwidget.enterEvent
        self.coordslabel.setMouseTracking(True)

        self.centerplot = None

        # Make a layout for the tab
        backwidget = QtGui.QWidget()
        self.layout.addWidget(backwidget)
        self.backlayout = QtGui.QHBoxLayout(backwidget)
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

        # Cache radial integration
        self.q, self.radialprofile = integration.radialintegrate(self.imgdata, self.experiment,
                                                                 mask=self.experiment.mask)


    def send1Dintegration(self):
        self.cache1Dintegration.emit(self.q, self.radialprofile)


    def mouseMoved(self, evt):
        """
        when the mouse is moved in the viewer, translate the crosshair, recalculate coordinates
        """
        pos = evt  ## using signal proxy turns original arguments into a tuple
        if self.viewbox.sceneBoundingRect().contains(pos):
            mousePoint = self.viewbox.mapSceneToView(pos)
            if (0 < mousePoint.x() < self.imgdata.shape[0]) & (
                            0 < mousePoint.y() < self.imgdata.shape[1]):  # within bounds
                #angstrom=QChar(0x00B5)
                if self.experiment.iscalibrated:
                    self.coordslabel.setText(u"<span style='font-size: 12pt;background-color:black;'>x=%0.1f,"
                                             u"   <span style=''>y=%0.1f</span>,   <span style=''>I=%0.0f</span>,"
                                         u"  q=%0.3f \u212B\u207B\u00B9,  q<sub>z</sub>=%0.3f \u212B\u207B\u00B9,"
                                         u"  q<sub>\u2225\u2225</sub>=%0.3f \u212B\u207B\u00B9</span>" % (
                                         mousePoint.x(),
                                         mousePoint.y(),
                                         self.imgdata[int(mousePoint.x()),
                                                      int(mousePoint.y())],
                                         pixel2q(mousePoint.x(),
                                                 mousePoint.y(),
                                                 self.experiment),
                                         pixel2q(None,
                                                 mousePoint.y(),
                                                 self.experiment),
                                         pixel2q(mousePoint.x(),
                                                 None,
                                                 self.experiment)))
                else:
                    self.coordslabel.setText(u"<span style='font-size: 12pt;background-color:black;'>x=%0.1f,"
                                             u"   <span style=''>y=%0.1f</span>,   <span style=''>I=%0.0f</span>,"
                                             u"  Calibration Required..." % (
                                                 mousePoint.x(),
                                                 mousePoint.y(),
                                                 self.imgdata[int(mousePoint.x()),
                                                              int(mousePoint.y())],
                                             ))

                self.coordslabel.setVisible(True)
            else:
                self.coordslabel.setVisible(False)
            self.vLine.setPos(mousePoint.x())
            self.hLine.setPos(mousePoint.y())
            self.parentwindow.qLine.setPos(pixel2q(mousePoint.x(), mousePoint.y(), self.experiment))

    def leaveEvent(self, evt):
        """
        hide crosshair and coordinates when mouse leaves viewer
        """
        self.hLine.setVisible(False)
        self.vLine.setVisible(False)
        self.coordslabel.setVisible(False)
        self.parentwindow.qLine.setVisible(False)

    def enterEvent(self, evt):
        """
        show crosshair and coordinates when mouse enters viewer
        """
        self.hLine.setVisible(True)
        self.vLine.setVisible(True)
        self.parentwindow.qLine.setVisible(True)


    def redrawimage(self):
        """
        redraws the diffraction image, checking drawing modes (log, symmetry, mask, cake)
        """
        islogintensity = self.parentwindow.ui.findChild(QtGui.QAction, 'actionLog_Intensity').isChecked()
        isradialsymmetry = self.parentwindow.ui.findChild(QtGui.QAction, 'actionRadial_Symmetry').isChecked()
        ismirrorsymmetry = self.parentwindow.ui.findChild(QtGui.QAction, 'actionMirror_Symmetry').isChecked()
        ismaskshown = self.parentwindow.ui.findChild(QtGui.QAction, 'actionShow_Mask').isChecked()
        iscake = self.parentwindow.ui.findChild(QtGui.QAction, 'actionCake').isChecked()
        isremesh = self.parentwindow.ui.findChild(QtGui.QAction, 'actionRemeshing').isChecked()
        img = self.imgdata.copy()
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
            marginmask = 1 - detectors.ALL_DETECTORS[self.experiment.getvalue('Detector')]().calc_mask().T
            imtest(marginmask)

            x, y = np.indices(img.shape)
            padmask = ((yshift < y) & (y < (yshift + img.shape[1])) & (xshift < x) & (x < (xshift + img.shape[0])))
            imtest(padmask)
            imtest(symimg * padmask * (1 - marginmask))
            img = img * marginmask + symimg * padmask * (1 - marginmask)

        elif ismirrorsymmetry:
            centery = self.experiment.getvalue('Center Y')
            symimg = np.fliplr(img.copy())
            imtest(symimg)
            yshift = -(img.shape[1] - 2 * centery)
            symimg = np.roll(symimg, int(yshift), axis=1)
            imtest(symimg)
            marginmask = 1 - detectors.ALL_DETECTORS[self.experiment.getvalue('Detector')]().calc_mask().T
            imtest(marginmask)

            x, y = np.indices(img.shape)
            padmask = ((yshift < y) & (y < (yshift + img.shape[1])))
            imtest(padmask)
            imtest(symimg * padmask * (1 - marginmask))
            img = img * marginmask + symimg * padmask * (1 - marginmask)

        if iscake:
            img, x, y = integration.cake(img, self.experiment)
        elif isremesh:
            img = remesh.remesh(self.path, self.experiment.getGeometry())

        if ismaskshown:
            self.maskimage.setImage(np.dstack((
                self.experiment.mask, np.zeros_like(self.experiment.mask), np.zeros_like(self.experiment.mask),
                self.experiment.mask)), opacity=.25)
        else:
            self.maskimage.clear()

        # When the log intensity button toggles, switch the log scaling on the image
        if islogintensity:
            img = (np.log(img * (img > 0) + (img < 1)))

        self.imageitem.setImage(img)
        #self.imageitem.setLookupTable(colormap.LUT)

    def linecut(self):
        """
        toggles the line cut
        """
        if self.parentwindow.ui.findChild(QtGui.QAction, 'actionLine_Cut').isChecked():
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

    def verticalcut(self):
        if self.parentwindow.ui.findChild(QtGui.QAction, 'actionVertical_Cut').isChecked():
            try:
                self.viewbox.removeItem(self.region)
            except AttributeError:
                print('Attribute error in verticalcut')
            self.region = pg.LinearRegionItem(orientation=pg.LinearRegionItem.Vertical, brush=pg.mkBrush('#00FFFF32'),
                                              bounds=[0, self.imgdata.shape[1]],
                                              values=[self.experiment.getvalue('Center X') - 10,
                                                      10 + self.experiment.getvalue('Center X')])
            for line in self.region.lines:
                line.setPen(pg.mkPen('#00FFFF'))
            self.region.sigRegionChangeFinished.connect(self.replot)
            self.viewbox.addItem(self.region)
        else:
            self.viewbox.removeItem(self.region)
            self.region = None
        self.replot()

    def horizontalcut(self):
        if self.parentwindow.ui.findChild(QtGui.QAction, 'actionHorizontal_Cut').isChecked():
            try:
                self.viewbox.removeItem(self.region)
            except AttributeError:
                print('Attribute error in horizontalcut')
            self.region = pg.LinearRegionItem(orientation=pg.LinearRegionItem.Horizontal, brush=pg.mkBrush('#00FFFF32'),
                                              bounds=[0, self.imgdata.shape[0]],
                                              values=[self.experiment.getvalue('Center Y') - 10,
                                                      10 + self.experiment.getvalue('Center Y')])
            for line in self.region.lines:
                line.setPen(pg.mkPen('#00FFFF'))
            self.region.sigRegionChangeFinished.connect(self.replot)
            self.viewbox.addItem(self.region)
        else:
            self.viewbox.removeItem(self.region)
            self.region = None
        self.replot()


    def removecosmics(self):
        c = cosmics.cosmicsimage(self.imgdata)
        c.run(maxiter=4)
        self.experiment.addtomask(c.mask)
        #self.maskoverlay()

    #@debug.timeit
    def findcenter(self):
        # Auto find the beam center
        [x, y] = center_approx.center_approx(self.imgdata, self.experiment)

        # Set the center in the experiment
        self.experiment.setvalue('Center X', x)
        self.experiment.setvalue('Center Y', y)
        self.drawcenter()
        self.replot()

    def drawcenter(self):
        # Mark the center
        self.centerplot = pg.ScatterPlotItem([self.experiment.getvalue('Center X')],
                                             [self.experiment.getvalue('Center Y')], pen=None, symbol='o')
        self.viewbox.addItem(self.centerplot)

    #@debug.timeit
    def calibrate(self):
        # Choose detector
        self.finddetector()

        self.findcenter()

        # x, y = np.indices(self.imgdata.shape)
        #r = np.sqrt((x - self.experiment.getvalue('Center X')) ** 2 + (y - self.experiment.getvalue('Center Y')) ** 2)
        #r = r.astype(np.int)
        #tbin = np.bincount(r.ravel(), self.imgdata.ravel())
        #nr = np.bincount(r.ravel(), self.experiment.mask.ravel())
        #with np.errstate(divide='ignore', invalid='ignore'):
        #    radialprofile = tbin / nr

        _, radialprofile = integration.radialintegrate(self.imgdata, self.experiment, mask=self.experiment.mask)

        # Find peak positions, they represent the radii
        # peaks = scipy.signal.find_peaks_cwt(np.nan_to_num(np.log(radialprofile + 3)), np.arange(1, 100))
        # np.set_printoptions(threshold=np.nan)
        # print('size', radialprofile.shape[0])
        peaks = np.array(peakfinding.findpeaks(np.arange(radialprofile.shape[0]), radialprofile)).T
        #print('after',PeakFinding.findpeaks(np.arange(radialprofile.__len__()),radialprofile)[0].shape)

        peaks = peaks[peaks[:, 1].argsort()[::-1]]

        #print peaks

        for peak in peaks:
            if peak[0] > 25 and not np.isinf(peak[1]):  ####This thresholds the minimum sdd which is acceptable
                bestpeak = peak[0]
                print peak
                break


        # Get the tallest peak
        #bestpeak = peaks[radialprofile[peaks].argmax()]

        # Calculate sample to detector distance for lowest q peak
        tth = 2 * np.arcsin(0.5 * self.experiment.getvalue('Wavelength') / 58.367e-10)
        tantth = np.tan(tth)
        sdd = bestpeak * self.experiment.getvalue('Pixel Size X') / tantth

        self.experiment.setvalue('Detector Distance', sdd)

        center_approx.refinecenter(self.imgdata, self.experiment)

        self.experiment.iscalibrated = True

        self.replot()
        # self.maskoverlay()

    def replot(self):
        self.parentwindow.integration.clear()

        self.replotprimary()
        self.parentwindow.qLine = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('#FFA500'))
        self.parentwindow.qLine.setVisible(False)
        self.parentwindow.integration.addItem(self.parentwindow.qLine)

    def replotprimary(self):
        cut = None

        if self.parentwindow.ui.findChild(QtGui.QAction, 'actionMultiPlot').isChecked():
            for tabtracker in self.parentwindow.ui.findChildren(imageTabTracker):
                if self.parentwindow.ui.findChild(QtGui.QTabWidget, 'tabWidget').currentWidget() is not tabtracker:
                    tabtracker.replotassecondary()

        if self.parentwindow.ui.findChild(QtGui.QAction, 'actionLine_Cut').isChecked():
            # regionbounds=self.region.getRegion()
            #cut = np.zeros_like(self.imgdata)
            #cut[regionbounds[0]:regionbounds[1],:]=1
            cut = self.region.getArrayRegion(self.imgdata, self.imageitem)

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
            if self.parentwindow.ui.findChild(QtGui.QAction, 'actionHorizontal_Cut').isChecked():
                regionbounds = self.region.getRegion()
                cut = np.zeros_like(self.imgdata)
                cut[:, regionbounds[0]:regionbounds[1]] = 1
            if self.parentwindow.ui.findChild(QtGui.QAction, 'actionVertical_Cut').isChecked():
                regionbounds = self.region.getRegion()
                cut = np.zeros_like(self.imgdata)
                cut[regionbounds[0]:regionbounds[1], :] = 1


            # Radial integration
            self.q, self.radialprofile = integration.radialintegrate(self.imgdata, self.experiment,
                                                                          mask=self.experiment.mask, cut=cut)
            self.cache1Dintegration.emit(self.q, self.radialprofile)
            ##############################################################################
            # Remi's peak finding
            # self.q / 10.0 is x
            # self.radialprofile is y
            # Find the peaks, and then plot them

            self.peaktooltip = peakfinding.peaktooltip(self.q, self.radialprofile, self.parentwindow.integration)

            # q, I, width, index = PeakFinding.findpeaks(self.q, self.radialprofile)

            #self.parentwindow.integration.plot(q, I, pen=None, symbol='o')

            ##############################################################################
            # Replot
            self.parentwindow.integration.plot(self.q, self.radialprofile)




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
            #print maskedarea.shape

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
            if detector.MAX_SHAPE == self.imgdata.shape[::-1]:
                detector = detector()
                mask = detector.calc_mask()
                self.experiment.addtomask(np.rot90(mask))
                self.experiment.setvalue('Pixel Size X', detector.pixel1)
                self.experiment.setvalue('Pixel Size Y', detector.pixel2)
                self.experiment.setvalue('Detector', name)
                return detector

    def exportimage(self):
        fabimg = edfimage.edfimage(np.rot90(self.imageitem.image))
        dialog = QtGui.QFileDialog(parent=self.parentwindow.ui, caption="blah", directory=os.path.dirname(self.path),
                                   filter=u"EDF (*.edf)")
        dialog.selectFile(os.path.basename(self.path))
        filename, _ = dialog.getSaveFileName()
        fabimg.write(filename)


class previewwidget(pg.GraphicsLayoutWidget):
    def __init__(self, model):
        super(previewwidget, self).__init__()
        self.model = model
        # self.setLayout(QHBoxLayout())
        #self.layout().setContentsMargins(0, 0, 0, 0)
        self.view = self.addViewBox(lockAspect=True)

        self.imageitem = pg.ImageItem()
        self.view.addItem(self.imageitem)
        self.imgdata = None
        # self.setMaximumHeight(100)
        # self.addItem(self.imageitem)

    def loaditem(self, index):
        path = self.model.filePath(index)
        self.imgdata, paras = loader.loadpath(path)
        self.imageitem.setImage(np.rot90(np.log(self.imgdata * (self.imgdata > 0) + (self.imgdata < 1)), 3),
                                autoLevels=True)


def imtest(image):
    if False:
        image = image * 255 / image.max()
        cv2.imshow('step?', cv2.resize(image.astype(np.uint8), (0, 0), fx=.2, fy=.2))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def pixel2q(x, y, experiment):
    # SWITCH TO PYFAI GEOMETRY

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