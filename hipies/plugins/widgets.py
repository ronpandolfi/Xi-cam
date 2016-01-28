# --coding: utf-8 --

from PySide import QtGui, QtCore
from PySide.QtCore import Qt
import numpy as np
import pyqtgraph as pg
from pipeline import loader, cosmics, integration, peakfinding, center_approx, variationoperators, pathtools
from hipies import config, ROI, globals, debugtools, toolbar
from fabio import edfimage
import os
import pyqtgraph.parametertree.parameterTypes as pTypes
from pyqtgraph.parametertree import Parameter, ParameterTree, ParameterItem, registerParameterType

class OOMTabItem(QtGui.QWidget):
    sigLoaded = QtCore.Signal()

    def __init__(self, itemclass=None, *args, **kwargs):
        """
        A collection of references that can be used to reconstruct an ojbect dynamically and dispose of it when unneeded
        :type paths: list[str]
        :type experiment: config.experiment
        :type parent: main.MyMainWindow
        :type operation:
        :return:
        """
        super(OOMTabItem, self).__init__()

        self.itemclass = itemclass
        self.args = args
        self.kwargs = kwargs

        self.isloaded = False
        self.setLayout(QtGui.QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)


    def load(self):
        """
        load this tab; rebuild the viewer
        """
        if not self.isloaded:
            if 'operation' in self.kwargs:
                if self.kwargs['operation'] is not None:
                    print self.kwargs['paths']
                    imgdata = [loader.loadimage(path) for path in self.kwargs['paths']]
                    imgdata = self.kwargs['operation'](imgdata)
                    dimg = loader.diffimage(filepath=self.kwargs['paths'][0], data=imgdata)
                    self.kwargs['dimg'] = dimg

            self.widget = self.itemclass(*self.args, **self.kwargs)

            self.layout().addWidget(self.widget)

            self.isloaded = True

            self.sigLoaded.emit()
            print 'Loaded:', self.itemclass


    def unload(self):
        """
        orphan the tab widget and queue them for deletion. Mwahahaha.
        """
        if self.isloaded:
            self.widget.deleteLater()
            self.widget = None

        self.isloaded = False


class dimgViewer(QtGui.QWidget):
    sigPlotQIntegration = QtCore.Signal(object)
    sigPlotChiIntegration = QtCore.Signal(object)


    def __init__(self, dimg=None, paths=None, plotwidget=None, toolbar=None, **kwargs):
        """
        A tab containing an imageview. Also manages functionality connected to a specific tab (masking/integration)
        :param imgdata:
        :param experiment:
        :return:
        """
        super(dimgViewer, self).__init__()

        self.region = None
        self.maskROI = None
        self.layout = QtGui.QStackedLayout(self)
        self.plotwidget = plotwidget
        self.toolbar = toolbar

        if type(paths) not in [list, None]:
            paths = [paths]

        self.paths = paths

        # Save image data and the experiment
        self.dimg = dimg

        if len(paths) == 1 and paths[0] is not None:
            self.dimg = loader.diffimage(filepath=paths[0])





        # For storing what action is active (mask/circle fit...)
        self.activeaction = None

        # Make an imageview for the image
        self.imgview = ImageView(self)
        self.imageitem = self.imgview.getImageItem()
        self.graphicslayoutwidget = self.imgview
        self.imgview.ui.roiBtn.setParent(None)

        self.viewbox = self.imageitem.getViewBox()
        # self.imgview.view.removeItem(self.imgview.roi)
        # self.imgview.roi.parent = None
        # self.imgview.roi.deleteLater()
        # self.imgview.view.removeItem(self.imgview.normRoi)       #Should uncomment?
        # self.imgview.normRoi.parent = None
        # self.imgview.normRoi.deleteLater()
        self.viewbox.invertY(False)
        reset = QtGui.QAction('Reset', self.imgview.getHistogramWidget().item.vb.menu)
        self.imgview.getHistogramWidget().item.vb.menu.addAction(reset)
        reset.triggered.connect(self.resetLUT)


        # self.imgview.getHistogramWidget().plot.setLogMode(True,False)


        # self.threads = dict()

        if self.plotwidget is not None:
            self.plotwidget.sigReplot.connect(self.replot)


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

        self.coordslabel = QtGui.QLabel(' ')
        self.coordslabel.setMinimumHeight(16)
        self.imgview.layout().addWidget(self.coordslabel)
        self.imgview.setStyleSheet("background-color: rgba(0,0,0,0%)")
        self.coordslabel.setAlignment(Qt.AlignHCenter | Qt.AlignBottom)
        self.coordslabel.setStyleSheet("background-color: rgba(0,0,0,0%)")
        self.graphicslayoutwidget.scene.sigMouseMoved.connect(self.mouseMoved)
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

        self.centerplot = pg.ScatterPlotItem()
        self.viewbox.addItem(self.centerplot)

        self.sgToverlay = pg.ScatterPlotItem()
        self.viewbox.addItem(self.sgToverlay)

        self.sgRoverlay = pg.ScatterPlotItem()
        self.viewbox.addItem(self.sgRoverlay)

        # Make a layout for the tab
        backwidget = QtGui.QWidget()
        self.layout.addWidget(backwidget)
        self.backlayout = QtGui.QHBoxLayout(backwidget)
        self.backlayout.setContentsMargins(0, 0, 0, 0)
        self.backlayout.addWidget(self.graphicslayoutwidget)

        # Add the Log Intensity check button to the context menu and wire up
        # self.imgview.buildMenu()
        # menu = self.viewbox.menu
        # self.actionLogIntensity = QAction('Log Intensity', menu, checkable=True)
        # self.actionLogIntensity.triggered.connect(self.logintensity)
        #menu.addAction(self.actionLogIntensity)
        #self.imgview.buildMenu()

        # Add a placeholder image item for the mask to the viewbox
        self.maskimage = pg.ImageItem(opacity=.25)
        self.viewbox.addItem(self.maskimage)

        self.sigPlotQIntegration.connect(self.plotqintegration)
        self.sigPlotChiIntegration.connect(self.plotchiintegration)

        # import ROI
        # self.arc=ROI.ArcROI((620.,29.),500.)
        # self.viewbox.addItem(self.arc)
        #print self.dimg.data
        #print self.imageitem


        #self.viewbox.addItem(pg.SpiralROI((0,0),1))

        try:
            energy = self.dimg.headers['Beamline Energy']
            if energy is not None:
                config.activeExperiment.setvalue('Energy', energy)
        except (AttributeError, TypeError, KeyError):
            print('Warning: Energy could not be determined from headers')


        # Cache radial integration
        if self.dimg is not None:
            if self.dimg.data is not None:
                self.redrawimage()

                #self.q, self.radialprofile = self.dimg.radialintegrate

                # Force cache the detector
                # _ = self.dimg.detector
                if not self.loadLUT():
                    self.resetLUT()

                if config.activeExperiment.iscalibrated:
                    self.replot()
                    self.drawcenter()

        self.imgview.getHistogramWidget().item.sigLevelChangeFinished.connect(self.cacheLUT)
        self.imgview.getHistogramWidget().item.gradient.sigGradientChangeFinished.connect(self.cacheLUT)

    def loadLUT(self):

        if globals.LUT is not None:
            hist = self.imgview.getHistogramWidget().item
            hist.setLevels(*globals.LUTlevels)
            hist.gradient.restoreState(globals.LUTstate)
            return True
        return False

    def resetLUT(self):

        print 'Levels:', self.imgview.getHistogramWidget().item.getLevels()
        # if self.imgview.getHistogramWidget().item.getLevels()==(0,1.):
        Lmax = np.nanmax(self.dimg.data)

        if self.toolbar.actionLog_Intensity.isChecked():
            self.imgview.getHistogramWidget().item.setLevels(
                np.log(max(np.nanmin(self.dimg.data * (self.dimg.data > 0)), 1)), np.log(Lmax))
        else:
            self.imgview.getHistogramWidget().item.setLevels(np.max(np.nanmin(self.dimg.data), 0), Lmax)
        print 'Levels set:', self.imgview.getHistogramWidget().item.getLevels()

    def cacheLUT(self):
        hist = self.imgview.getHistogramWidget().item
        globals.LUTlevels = hist.getLevels()
        globals.LUTstate = hist.gradient.saveState()
        globals.LUT = hist.getLookupTable(img=self.imageitem.image)

    # def send1Dintegration(self):
    # self.cache1Dintegration.emit(self.q, self.radialprofile)

    def removeROI(self, evt):

        # evt.scene().removeItem(evt)
        self.viewbox.removeItem(evt)
        self.replot()
        # evt.deleteLater()

        # self.viewbox.scene().removeItem(evt)

    @property
    def isChiPlot(self):
        return self.plotwidget.currentIndex()

    def mouseMoved(self, evt):
        """
        when the mouse is moved in the viewer, translate the crosshair, recalculate coordinates
        """
        pos = evt  ## using signal proxy turns original arguments into a tuple
        if self.viewbox.sceneBoundingRect().contains(pos):
            mousePoint = self.viewbox.mapSceneToView(pos)
            self.vLine.setPos(mousePoint.x())
            self.hLine.setPos(mousePoint.y())
            if (0 < mousePoint.x() < self.imageitem.image.shape[0]) & (
                            0 < mousePoint.y() < self.imageitem.image.shape[1]):  # within bounds
                # angstrom=QChar(0x00B5)
                if config.activeExperiment.iscalibrated:
                    x = mousePoint.x()
                    y = mousePoint.y()

                    iscake = self.toolbar.actionCake.isChecked()
                    isremesh = self.toolbar.actionRemeshing.isChecked()

                    if iscake:
                        data = self.dimg.cake
                    elif isremesh:
                        data = self.dimg.remesh
                    else:
                        data = self.dimg.data

                    # if iscake:
                    # q = pixel2cake(x, y, self.dimg)
                    #
                    # elif isremesh:
                    # return
                    # else:
                    #     q = pixel2q(x, y, self.dimg.experiment)
                    #print x,y,self.dimg.data[int(x),int(y)],self.getq(x,y),self.getq(None,y),self.getq(x,None,),np.sqrt((x - self.dimg.experiment.center[0]) ** 2 + (y - self.dimg.experiment.center[1]) ** 2)
                    self.coordslabel.setText(u"<div style='font-size: 12pt;background-color:#111111;'>x=%0.1f,"
                                             u"   <span style=''>y=%0.1f</span>,   <span style=''>I=%0.0f</span>,"
                                             u"  q=%0.3f \u212B\u207B\u00B9,  q<sub>z</sub>=%0.3f \u212B\u207B\u00B9,"
                                             u"  q<sub>\u2225\u2225</sub>=%0.3f \u212B\u207B\u00B9,"
                                             u"  d=%0.3f nm,"
                                             u"  \u03B8=%.2f</div>" % (
                                                 x,
                                                 y,
                                                 data[int(x),
                                                      int(y)],
                                                 self.getq(x, y),
                                                 self.getq(x, y, 'z'),
                                                 self.getq(x, y, 'parallel'),
                                                 2 * np.pi / self.getq(x, y) / 10,
                                                 360. / (2 * np.pi) * np.arctan2(self.getq(x, y, 'z'),
                                                                                 self.getq(x, y, 'parallel'))))
                    # np.sqrt((x - self.dimg.experiment.center[0]) ** 2 + (
                    #y - self.dimg.experiment.center[1]) ** 2)))
                    #,  r=%0.1f
                    if hasattr(self.plotwidget, 'qintegration'):
                        self.plotwidget.qintegration.qLine.setPos(self.getq(mousePoint.x(), mousePoint.y()))
                        self.plotwidget.qintegration.qLine.show()
                else:
                    self.coordslabel.setText(u"<div style='font-size: 12pt;background-color:#111111;'>x=%0.1f,"
                                             u"   <span style=''>y=%0.1f</span>,   <span style=''>I=%0.0f</span>,"
                                             u"  Calibration Required...</div>" % (
                                                 mousePoint.x(),
                                                 mousePoint.y(),
                                                 self.dimg.data[int(mousePoint.x()),
                                                                int(mousePoint.y())],
                                             ))

                    # self.coordslabel.setVisible(True)

            else:
                self.coordslabel.setText(u"<div style='font-size:12pt;background-color:#111111;'>&nbsp;</div>")
                if hasattr(self.plotwidget, 'qintegration'):
                    self.plotwidget.qintegration.qLine.hide()


    def getq(self, x, y, mode=None):
        iscake = self.toolbar.actionCake.isChecked()
        isremesh = self.toolbar.actionRemeshing.isChecked()

        if iscake:
            cakeq = self.dimg.cakeqx
            cakechi = self.dimg.cakeqy
            if mode is not None:
                if mode == 'parallel':

                    return cakeq[y] * np.sin(np.radians(cakechi[x])) / 10.
                elif mode == 'z':

                    return cakeq[y] * np.cos(np.radians(cakechi[x])) / 10.
            else:
                return cakeq[y] / 10.

        elif isremesh:
            remeshqpar = self.dimg.remeshqx
            remeshqz = self.dimg.remeshqy
            if mode is not None:
                if mode == 'parallel':
                    return remeshqpar[x, y] / 10.
                elif mode == 'z':
                    return -remeshqz[x, y] / 10.
            else:
                return np.sqrt(remeshqz[x, y] ** 2 + remeshqpar[x, y] ** 2) / 10.

        else:
            center = config.activeExperiment.center
            if mode is not None:
                if mode == 'z':
                    x = center[0]
                if mode == 'parallel':
                    y = center[1]

            r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
            theta = np.arctan2(r * config.activeExperiment.getvalue('Pixel Size X'),
                               config.activeExperiment.getvalue('Detector Distance'))
            wavelength = config.activeExperiment.getvalue('Wavelength')
            return 4 * np.pi / wavelength * np.sin(theta / 2) * 1e-10

    def leaveEvent(self, evt):
        """
        hide crosshair and coordinates when mouse leaves viewer
        """
        self.hLine.setVisible(False)
        self.vLine.setVisible(False)
        # self.coordslabel.setVisible(False)
        if hasattr(self.plotwidget, 'qintegration'):
            self.plotwidget.qintegration.qLine.setVisible(False)

    def enterEvent(self, evt):
        """
        show crosshair and coordinates when mouse enters viewer
        """
        self.hLine.setVisible(True)
        self.vLine.setVisible(True)
        if hasattr(self.plotwidget, 'qintegration'):
            self.plotwidget.qintegration.qLine.setVisible(True)


    def redrawimageLowRes(self):
        self.redrawimage(forcelow=True)


    def redrawimage(self, returnimg=False):
        """
        redraws the diffraction image, checking drawing modes (log, symmetry, mask, cake)
        """
        toolbar = self.toolbar

        img = self.dimg.data
        scale = 1

        islogintensity = toolbar.actionLog_Intensity.isChecked()
        isradialsymmetry = toolbar.actionRadial_Symmetry.isChecked()
        ismirrorsymmetry = toolbar.actionMirror_Symmetry.isChecked()
        ismaskshown = toolbar.actionShow_Mask.isChecked()
        iscake = toolbar.actionCake.isChecked()
        isremesh = toolbar.actionRemeshing.isChecked()
        # img = self.dimg.data.copy()
        # if forcelow:
        # img = self.dimg.thumbnail.copy()
        # scale = 10
        # else:




        if isradialsymmetry:
            centerx = config.activeExperiment.center[0]
            centery = config.activeExperiment.center[1]
            symimg = np.rot90(img.copy(), 2)
            # imtest(symimg)
            xshift = -(img.shape[0] - 2 * centerx)
            yshift = -(img.shape[1] - 2 * centery)
            symimg = np.roll(symimg, int(xshift), axis=0)
            symimg = np.roll(symimg, int(yshift), axis=1)
            # imtest(symimg)
            marginmask = config.activeExperiment.mask
            #imtest(marginmask)

            x, y = np.indices(img.shape)
            padmask = ((yshift < y) & (y < (yshift + img.shape[1])) & (xshift < x) & (x < (xshift + img.shape[0])))
            # imtest(padmask)
            #imtest(symimg * padmask * (1 - marginmask))
            img = img * marginmask + symimg * padmask * (1 - marginmask)

        elif ismirrorsymmetry:
            centery = config.activeExperiment.getvalue('Center Y')
            symimg = np.fliplr(img.copy())
            #imtest(symimg)
            yshift = -(img.shape[1] - 2 * centery)
            symimg = np.roll(symimg, int(yshift), axis=1)
            #imtest(symimg)
            marginmask = 1 - config.activeExperiment.mask
            #imtest(marginmask)

            x, y = np.indices(img.shape)
            padmask = ((yshift < y) & (y < (yshift + img.shape[1])))
            # imtest(padmask)
            #imtest(symimg * padmask * (1 - marginmask))
            img = img * marginmask + symimg * padmask * (1 - marginmask)

        mask = config.activeExperiment.mask

        if self.iscake:
            img = self.dimg.cake
            # print self.dimg.cakeqx
            #print self.dimg.cakeqy

        elif self.isremesh:
            img = self.dimg.remesh
            # print self.dimg.remeshqx
            #print self.dimg.remeshqy

        if self.iscake:
            if self.centerplot is not None:
                self.centerplot.clear()
        else:
            self.drawcenter()

        if self.ismaskshown:
            self.maskoverlay()
        else:
            self.maskimage.clear()

        # When the log intensity button toggles, switch the log scaling on the image
        if islogintensity:
            img = (np.log(img * (img > 0) + (img < 1)))

        if returnimg:
            return img
        else:
            self.imageitem.setImage(img, scale=scale)

        if not iscake and not isremesh:
            self.imageitem.setRect(QtCore.QRect(0, 0, self.dimg.data.shape[0], self.dimg.data.shape[1]))


    def getcenter(self):
        if self.isremesh:
            remeshqpar = self.dimg.remeshqy
            remeshqz = self.dimg.remeshqx
            q = remeshqpar ** 2 + remeshqz ** 2
            center = np.where(q == q.min())
            print 'remeshcenter:', center
            return zip(*center)[0]
        else:
            return config.activeExperiment.center

    @property
    def islogintensity(self):
        return self.toolbar.actionLog_Intensity.isChecked()

    @property
    def isradialsymmetry(self):
        return self.toolbar.actionRadial_Symmetry.isChecked()

    @property
    def ismirrorsymmetry(self):
        return self.toolbar.actionMirror_Symmetry.isChecked()

    @property
    def ismaskshown(self):
        return self.toolbar.actionShow_Mask.isChecked()

    @property
    def iscake(self):
        return self.toolbar.actionCake.isChecked()

    @property
    def isremesh(self):
        return self.toolbar.actionRemeshing.isChecked()

    def arccut(self):


        arc = ROI.ArcROI(self.getcenter(), 500)
        arc.sigRegionChangeFinished.connect(self.replot)
        self.viewbox.addItem(arc)
        self.replot()
        arc.sigRemoveRequested.connect(self.removeROI)


    def linecut(self):
        """
        toggles the line cut
        """
        # self.viewbox.removeItem(self.region)
        # self.parentwindow.difftoolbar.actionVertical_Cut.setChecked(False)
        # self.parentwindow.difftoolbar.actionHorizontal_Cut.setChecked(False)
        # if self.parentwindow.difftoolbar.actionLine_Cut.isChecked():
        region = ROI.LineROI(
            self.getcenter(),
            [self.getcenter()[0], -self.dimg.data.shape[0]], 5, removable=True)
        region.sigRemoveRequested.connect(self.removeROI)
        self.viewbox.addItem(region)
        self.replot()
        region.sigRegionChangeFinished.connect(self.replot)
        # else:
        # #self.viewbox.removeItem(self.region)
        # self.region = None
        # self.replot()

    def verticalcut(self):
        # self.viewbox.removeItem(self.region)
        # self.parentwindow.difftoolbar.actionLine_Cut.setChecked(False)
        # self.parentwindow.difftoolbar.actionHorizontal_Cut.setChecked(False)
        # if self.parentwindow.difftoolbar.actionVertical_Cut.isChecked():
        # try:
        # self.viewbox.removeItem(self.region)
        # except AttributeError:
        #         print('Attribute error in verticalcut')
        region = ROI.LinearRegionItem(orientation=pg.LinearRegionItem.Vertical, brush=pg.mkBrush('#00FFFF32'),
                                      bounds=[0, self.dimg.data.shape[1]],
                                      values=[self.getcenter()[0] - 10,
                                              10 + self.getcenter()[0]])
        for line in region.lines:
            line.setPen(pg.mkPen('#00FFFF'))
        region.sigRegionChangeFinished.connect(self.replot)
        region.sigRemoveRequested.connect(self.removeROI)
        self.viewbox.addItem(region)
        self.replot()
        # else:
        # #self.viewbox.removeItem(self.region)
        #     self.region = None
        # self.replot()

    def horizontalcut(self):
        # self.parentwindow.difftoolbar.actionVertical_Cut.setChecked(False)
        # self.parentwindow.difftoolbar.actionLine_Cut.setChecked(False)
        # self.viewbox.removeItem(self.region)
        # if self.parentwindow.difftoolbar.actionHorizontal_Cut.isChecked():
        # try:
        # self.viewbox.removeItem(self.region)
        # except AttributeError:
        #         print('Attribute error in horizontalcut')
        region = ROI.LinearRegionItem(orientation=pg.LinearRegionItem.Horizontal, brush=pg.mkBrush('#00FFFF32'),
                                      bounds=[0, self.dimg.data.shape[0]],
                                      values=[10 - self.getcenter()[1],
                                              10 + self.getcenter()[1]])
        for line in region.lines:
            line.setPen(pg.mkPen('#00FFFF'))
        region.sigRegionChangeFinished.connect(self.replot)
        region.sigRemoveRequested.connect(self.removeROI)
        self.viewbox.addItem(region)

        if self.iscake:
            self.plotwidget.plotTabs.setCurrentIndex(1)

        self.replot()
        # else:
        # #self.viewbox.removeItem(self.region)
        #     self.region = None
        # self.replot()


    def removecosmics(self):
        c = cosmics.cosmicsimage(self.dimg.data)
        c.run(maxiter=4)
        config.activeExperiment.addtomask(c.mask)
        # self.maskoverlay()

    @debugtools.timeit
    def findcenter(self, skipdraw=False):
        # Auto find the beam center
        self.dimg.findcenter()
        if not skipdraw:
            self.drawcenter()
            self.replot()

    def drawcenter(self):

        center = self.getcenter()
        # Mark the center
        if self.centerplot is not None: self.centerplot.clear()
        self.centerplot.setData([center[0]], [center[1]], pen=None, symbol='o',
                                brush=pg.mkBrush('#FFA500'))
        # self.viewbox.addItem(self.centerplot)

        # Move Arc ROIs to center


        for item in self.viewbox.addedItems:
            #print item
            if issubclass(type(item), ROI.ArcROI):
                print 'Moving ArcROI:', item

                print 'Current pos:', item.state['pos']
                print 'New pos:', center
                item.setPos(center)

    @debugtools.timeit
    def calibrate(self):
        if self.dimg.data is None:
            return
        config.activeExperiment.iscalibrated = False

        # Force cache the detector
        _ = self.dimg.detector

        self.findcenter(skipdraw=True)
        print 'center?:', config.activeExperiment.center

        radialprofile = integration.pixel_2Dintegrate(self.dimg)

        peaks = np.array(peakfinding.findpeaks(np.arange(len(radialprofile)), radialprofile)).T

        peaks = peaks[peaks[:, 1].argsort()[::-1]]

        for peak in peaks:
            if peak[0] > 25 and not np.isinf(peak[1]):  ####This thresholds the minimum sdd which is acceptable
                bestpeak = peak[0]
                # print peak
                break

        # Calculate sample to detector distance for lowest q peak
        tth = 2 * np.arcsin(0.5 * config.activeExperiment.getvalue('Wavelength') / 58.367e-10)
        tantth = np.tan(tth)
        sdd = bestpeak * config.activeExperiment.getvalue('Pixel Size X') / tantth

        print 'Best AgB peak gives sdd: ' + str(sdd)

        config.activeExperiment.setvalue('Detector Distance', sdd)

        self.refinecenter()

        config.activeExperiment.iscalibrated = True

        self.replot()

    @debugtools.timeit
    def refinecenter(self):
        # Force cache the detector
        # _=self.dimg.detector

        cen = center_approx.refinecenter(self.dimg)
        config.activeExperiment.center = cen
        self.drawcenter()

    def replot(self):


        if self.plotwidget.currentIndex() == 0:
            self.plotwidget.qintegration.clear()

            # self.replotprimary()
            self.plotwidget.qintegration.qLine = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('#FFA500'))
            self.plotwidget.qintegration.qLine.setVisible(False)
            self.plotwidget.qintegration.addItem(self.plotwidget.qintegration.qLine)
            self.replotq()
        elif self.plotwidget.currentIndex() == 1:
            self.plotwidget.chiintegration.clear()
            self.plotwidget.chiintegration.chiLine = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('#FFA500'))
            self.plotwidget.chiintegration.chiLine.setVisible(False)
            self.plotwidget.chiintegration.addItem(self.plotwidget.chiintegration.chiLine)
            self.replotchi()

    def replotchi(self):
        if not config.activeExperiment.iscalibrated:
            return None

        cut = None

        # if self.toolbar.actionMultiPlot.isChecked():
        # for tab in self.parent().centerwidget.findChildren(OOMTabItem):
        # if self.parent().centerwidget.currentWidget() is not tab:
        #             tab.replotassecondary()

        if self.iscake:
            data = self.dimg.cake
        elif self.isremesh:
            data = self.dimg.remesh
        else:
            data = self.dimg.data

        ai = config.activeExperiment.getAI().getPyFAI()
        globals.pool.apply_async(integration.chiintegratepyFAI,
                                 args=(self.dimg.data, self.dimg.mask, ai, self.iscake ),
                                 callback=self.chiintegrationrelay)
        # pipeline.integration.chiintegratepyFAI(self.dimg.data, self.dimg.mask, ai, precaked=self.iscake)

        for roi in self.viewbox.addedItems:
            try:
                if hasattr(roi, 'isdeleting'):
                    if not roi.isdeleting:
                        print type(roi)
                        cut = None
                        if issubclass(type(roi), pg.ROI) or issubclass(type(roi), pg.LinearRegionItem):
                            cut = (roi.getArrayRegion(np.ones_like(data), self.imageitem)).T
                            print 'Cut:', cut.shape







                        # elif type(roi) is pg.LineSegmentROI:
                        #
                        #
                        # cut = self.region.getArrayRegion(data, self.imageitem)
                        #
                        #     x = np.linspace(self.viewbox.mapSceneToView(self.region.getSceneHandlePositions(0)[1]).x(),
                        #                     self.viewbox.mapSceneToView(self.region.getSceneHandlePositions(1)[1]).x(),
                        #                     cut.__len__())
                        #     y = np.linspace(self.viewbox.mapSceneToView(self.region.getSceneHandlePositions(0)[1]).y(),
                        #                     self.viewbox.mapSceneToView(self.region.getSceneHandlePositions(1)[1]).y(),
                        #                     cut.__len__())
                        #
                        #     q = pixel2q(x, y, self.dimg.experiment)
                        #     qmiddle = q.argmin()
                        #     leftq = -q[0:qmiddle]
                        #     rightq = q[qmiddle:]
                        #
                        #     if leftq.__len__() > 1: self.parentwindow.qintegration.plot(leftq, cut[:qmiddle])
                        #     if rightq.__len__() > 1: self.parentwindow.qintegration.plot(rightq, cut[qmiddle:])

                        # elif type(roi) is ROI.LinearRegionItem:
                        #     if roi.orientation is pg.LinearRegionItem.Horizontal:
                        #         regionbounds = roi.getRegion()
                        #         cut = np.zeros_like(data)
                        #         cut[:, regionbounds[0]:regionbounds[1]] = 1
                        #     elif roi.orientation is pg.LinearRegionItem.Vertical:
                        #         regionbounds = roi.getRegion()
                        #         cut = np.zeros_like(data)
                        #         cut[regionbounds[0]:regionbounds[1], :] = 1
                        #
                        #     else:
                        #         print debugtools.frustration()

                        if cut is not None:
                            if self.iscake:

                                ma = np.ma.masked_array(data, mask=1 - (cut * self.dimg.cakemask))
                                chi = self.dimg.cakeqy
                                I = np.ma.average(ma, axis=1)
                                I = np.trim_zeros(I, 'b')
                                chi = chi[:len(I)]
                                I = np.trim_zeros(I, 'f')
                                chi = chi[-len(I):]
                                self.plotchiintegration([chi, I, [0, 255, 255]])
                            else:

                                ai = config.activeExperiment.getAI().getPyFAI()
                                self.pool.apply_async(integration.chiintegratepyFAI,
                                                      args=(self.dimg.data, self.dimg.mask, ai, self.iscake, cut,
                                                            [0, 255, 255]),
                                                      callback=self.chiintegrationrelay)



                    else:
                        self.viewbox.removeItem(roi)
            except Exception as ex:
                print 'Warning: error displaying ROI integration.'
                print ex.message

    def replotq(self):
        if not config.activeExperiment.iscalibrated:
            return None

        cut = None

        # if self.toolbar.actionMultiPlot.isChecked():
        # for tabtracker in self.parent().findChildren(OOMTabItem):
        # if self.parent().findChild(QtGui.QTabWidget, 'tabWidget').currentWidget() is not tabtracker:
        #             tabtracker.replotassecondary()

        if self.iscake:
            data = self.dimg.cake
        elif self.isremesh:
            data = self.dimg.remesh
        else:
            data = self.dimg.data
        ai = config.activeExperiment.getAI().getPyFAI()
        print 'centeroverride:', [c * config.activeExperiment.getvalue('Pixel Size X') for c in
                                  self.getcenter()[::-1]]
        print self.getcenter()

        globals.pool.apply_async(integration.radialintegratepyFAI,
                                 args=((self.dimg.data if not self.isremesh else data),
                                       (self.dimg.mask if not self.isremesh else self.dimg.remeshmask), ai, None, None,
                                       [c * config.activeExperiment.getvalue('Pixel Size X') for c in
                                        self.getcenter()[::-1]]),
                                 callback=self.qintegrationrelay)

        for roi in self.viewbox.addedItems:
            try:
                if hasattr(roi, 'isdeleting'):
                    if not roi.isdeleting:
                        print type(roi)
                        cut = None
                        if issubclass(type(roi), pg.ROI) or issubclass(type(roi), pg.LinearRegionItem):

                            cut = (roi.getArrayRegion(np.ones_like(data), self.imageitem)).T
                            print 'Cut:', cut.shape







                        elif type(roi) is pg.LineSegmentROI:


                            cut = self.region.getArrayRegion(data, self.imageitem)

                            x = np.linspace(self.viewbox.mapSceneToView(self.region.getSceneHandlePositions(0)[1]).x(),
                                            self.viewbox.mapSceneToView(self.region.getSceneHandlePositions(1)[1]).x(),
                                            cut.__len__())
                            y = np.linspace(self.viewbox.mapSceneToView(self.region.getSceneHandlePositions(0)[1]).y(),
                                            self.viewbox.mapSceneToView(self.region.getSceneHandlePositions(1)[1]).y(),
                                            cut.__len__())

                            q = pixel2q(x, y, self.dimg.experiment)
                            qmiddle = q.argmin()
                            leftq = -q[0:qmiddle]
                            rightq = q[qmiddle:]

                            if leftq.__len__() > 1: self.plotwidget.qintegration.plot(leftq, cut[:qmiddle])
                            if rightq.__len__() > 1: self.plotwidget.qintegration.plot(rightq, cut[qmiddle:])

                        # elif type(roi) is ROI.LinearRegionItem:
                        #     if roi.orientation is pg.LinearRegionItem.Horizontal:
                        #         regionbounds = roi.getRegion()
                        #         cut = np.zeros_like(data)
                        #         cut[:, regionbounds[0]:regionbounds[1]] = 1
                        #     elif roi.orientation is pg.LinearRegionItem.Vertical:
                        #         regionbounds = roi.getRegion()
                        #         cut = np.zeros_like(data)
                        #         cut[regionbounds[0]:regionbounds[1], :] = 1
                        #
                        #     else:
                        #         print debugtools.frustration()

                        if cut is not None:
                            if self.iscake:

                                ma = np.ma.masked_array(data, mask=1 - (cut * self.dimg.cakemask))
                                q = self.dimg.cakeqx / 10.
                                I = np.ma.average(ma, axis=0)
                                I = np.trim_zeros(I, 'b')
                                q = q[:len(I)]
                                I = np.trim_zeros(I, 'f')
                                q = q[-len(I):]
                                self.plotqintegration([q, I, [0, 255, 255]])
                            else:

                                ai = config.activeExperiment.getAI().getPyFAI()
                                globals.pool.apply_async(integration.radialintegratepyFAI, args=(
                                    (self.dimg.data if not self.isremesh else data),
                                    (self.dimg.mask if not self.isremesh else self.dimg.remeshmask), ai, cut,
                                    [0, 255, 255], [c * config.activeExperiment.getvalue('Pixel Size X') for c in
                                                    self.getcenter()[::-1]]),
                                                         callback=self.qintegrationrelay)


                    else:
                        self.viewbox.removeItem(roi)
            except Exception as ex:
                print 'Warning: error displaying ROI integration.'
                print ex.message

    def qintegrationrelay(self, *args, **kwargs):
        self.sigPlotQIntegration.emit(*args, **kwargs)


    def plotqintegration(self, result):
        (q, radialprofile, color) = result
        if color is None:
            color = [255, 255, 255]
        # cyan:[0, 255, 255]
        curve = self.plotwidget.qintegration.plot(q, radialprofile, pen=pg.mkPen(color=color))
        curve.setZValue(3 * 255 - sum(color))
        self.plotwidget.qintegration.update()

    def chiintegrationrelay(self, *args, **kwargs):
        print args
        self.sigPlotChiIntegration.emit(*args, **kwargs)

    def plotchiintegration(self, result):
        (chi, chiprofile, color) = result
        if color is None:
            color = [255, 255, 255]
        # cyan:[0, 255, 255]
        curve = self.plotwidget.chiintegration.plot(chi, chiprofile, pen=pg.mkPen(color=color))
        curve.setZValue(3 * 255 - sum(color))
        self.plotwidget.chiintegration.update()


    def polymask(self):
        maskroi = None
        for roi in self.viewbox.addedItems:
            print type(roi)
            if type(roi) is pg.PolyLineROI:
                maskroi = roi

        if maskroi is None:

            # Start with a box around the center
            left = config.activeExperiment.getvalue('Center X') - 100
            right = config.activeExperiment.getvalue('Center X') + 100
            up = config.activeExperiment.getvalue('Center Y') - 100
            down = config.activeExperiment.getvalue('Center Y') + 100

            # Add ROI item to the image
            self.maskROI = pg.PolyLineROI([[left, up], [left, down], [right, down], [right, up]], pen=(6, 9),
                                          closed=True)
            self.viewbox.addItem(self.maskROI)

            # Override the ROI's function to check if any points will be moved outside the boundary; False prevents move
            def checkPointMove(handle, pos, modifiers):
                p = self.viewbox.mapToView(pos)
                if 0 < p.y() < self.dimg.data.shape[0] and 0 < p.x() < self.dimg.data.shape[1]:
                    return True
                else:
                    return False

            self.maskROI.checkPointMove = checkPointMove

        else:  # If the mask is completed
            self.activeaction = None

            # Get the region of the image that was selected; unforunately the region is trimmed
            maskedarea = self.maskROI.getArrayRegion(np.ones_like(self.dimg.data), self.imageitem,
                                                     returnMappedCoords=True)  # levels=(0, arr.max()
            # print maskedarea.shape

            # Decide how much to left and top pad based on the ROI bounding rectangle
            boundrect = self.viewbox.itemBoundingRect(self.maskROI)
            leftpad = max(boundrect.x(), 0)
            toppad = max(boundrect.y(), 0)

            print 'Leftpad:', leftpad
            print 'Rightpad:', toppad

            # Pad the mask so it has the same shape as the image
            maskedarea = np.pad(maskedarea, ((int(leftpad), 0), (int(toppad), 0)), mode='constant')
            maskedarea = np.pad(maskedarea, (
                (0, self.dimg.data.shape[0] - maskedarea.shape[0]), (0, self.dimg.data.shape[1] - maskedarea.shape[1])),
                                mode='constant')

            # Add the masked area to the active mask
            config.activeExperiment.addtomask(1 - maskedarea)
            self.dimg.invalidatecache()

            # Draw the overlay
            # self.maskoverlay()

            # Remove the ROI
            self.viewbox.removeItem(self.maskROI)

            # Redraw the mask
            self.maskoverlay()

            # Redo the integration
            self.replot()


    def maskoverlay(self):
        if self.iscake:
            mask = self.dimg.cakemask
        elif self.isremesh:
            mask = self.dimg.remeshmask
        else:
            mask = config.activeExperiment.mask

        # Draw the mask as a red channel image with an alpha mask
        print 'maskmax:', np.max(mask) * 1.0
        invmask = 1 - mask
        self.maskimage.setImage(
            np.dstack((invmask, np.zeros_like(invmask), np.zeros_like(invmask), invmask)).astype(np.int),
            opacity=.25)


    # def finddetector(self):
    # # detector, mask = pipeline.loader.finddetector(self.imgdata)
    # if self.path is not None:
    # name, mask, detector = pipeline.loader.finddetectorbyfilename(self.path)
    #     else:
    #         name, mask, detector = pipeline.loader.finddetector(self.imgdata)
    #     if detector is not None:
    #         # name = detector.get_name()
    #         if mask is not None:
    #             self.experiment.addtomask(np.rot90(mask))
    #         self.experiment.setvalue('Pixel Size X', detector.pixel1)
    #         self.experiment.setvalue('Pixel Size Y', detector.pixel2)
    #         self.experiment.setvalue('Detector', name)
    #     return detector


    def exportimage(self):
        fabimg = edfimage.edfimage(np.rot90(self.imageitem.image))
        dialog = QtGui.QFileDialog(parent=self, caption="blah", directory=os.path.dirname(self.path),
                                   filter=u"EDF (*.edf)")
        dialog.selectFile(os.path.basename(self.path))
        filename, _ = dialog.getSaveFileName()
        fabimg.write(filename)

    def drawsgoverlay(self, peakoverlay):
        self.viewbox.addItem(peakoverlay)
        peakoverlay.enable(self.viewbox)


class timelineViewer(dimgViewer):
    def __init__(self, simg=None, files=None, toolbar=None):
        self.variation = dict()
        self.toolbar = toolbar

        if simg is None:
            simg = loader.imageseries(paths=files)

        self.simg = simg
        dimg = simg.first()

        self.operationindex = 0

        super(timelineViewer, self).__init__(dimg, toolbar=toolbar)

        img = np.array(self.simg.thumbs)
        img = (np.log(img * (img > 0) + (img < 1)))

        self.imgview.setImage(img, xvals=self.simg.xvals)

        self.imageitem.sigImageChanged.connect(self.setscale)

        # self.paths = dict(zip(range(len(paths)), sorted(paths)))
        # self.setvariationmode(0)
        # self.gotomax()
        # self.imgview.timeLine.sigPositionChanged.disconnect(self.imgview.timeLineChanged)

        self.highresimgitem = pg.ImageItem()

        self.viewbox.addItem(self.highresimgitem)
        self.highresimgitem.hide()

        self.imgview.timeLine.sigPositionChangeFinished.connect(self.drawframeoverlay)
        self.imgview.timeLine.sigDragged.connect(self.hideoverlay)
        self.imgview.sigKeyRelease.connect(self.drawframeoverlay)

        self.drawframeoverlay()
        self.imgview.autoRange()

        timelineplot = self.imgview.getRoiPlot()
        self.timeline = timelineplot.getPlotItem()
        self.timeline.showAxis('left', False)
        self.timeline.showAxis('bottom', False)
        self.timeline.showAxis('top', True)
        self.timeline.showGrid(x=True)

        self.timeline.getViewBox().setMouseEnabled(x=False, y=True)
        # self.timeline.setLabel('bottom', u'Frame #', '')

        # self.timeline.getViewBox().buildMenu()
        menu = self.timeline.getViewBox().menu
        operationcombo = QtGui.QComboBox()
        operationcombo.setObjectName('operationcombo')
        operationcombo.addItems(variationoperators.operations.keys())
        operationcombo.currentIndexChanged.connect(self.setvariationmode)
        opwidgetaction = QtGui.QWidgetAction(menu)
        opwidgetaction.setDefaultWidget(operationcombo)
        #need to connect it
        menu.addAction(opwidgetaction)


        # self.imgview.getHistogramWidget().item.setImageItem(self.highresimgitem)
        #self.imgview.getHistogramWidget().item.sigLevelChangeFinished.connect(self.updatelowresLUT)


        self.rescan()

    def processtimeline(self):
        self.rescan()
        self.toolbar.actionProcess.setChecked(False)

    def aborttimeline(self):
        pass

    def plottimeline(self, t, V, color=[255, 255, 255]):
        pass

    def drawframeoverlay(self):
        self.scale = 1
        self.dimg = self.simg.getDiffImage(round(self.imgview.timeLine.getXPos()))
        self.imgview.imageItem.updateImage(self.redrawimage(returnimg=True), noscale=True)


    def updatelowresLUT(self):

        self.imageitem.setLookupTable(self.imgview.getHistogramWidget().item.getLookupTable)


    def hideoverlay(self):
        self.scale = 5

    def setscale(self):
        self.imageitem.resetTransform()
        self.imageitem.scale(self.scale, self.scale)

    def showlowres(self):
        # self.imgview.setImage(np.repeat(np.repeat(np.array(self.simg.thumbs.values()), 10, axis=0), 10, axis=1),
        # xvals=self.simg.xvals)
        self.imgview.setImage(np.array(self.simg.thumbs.values()), xvals=self.simg.xvals)


    def reduce(self):
        pass
        # self.skipframes = (self.variationy[0:-1] / self.variationy[1:]) > 0.1

    def appendimage(self, d, paths):
        paths = [os.path.join(d, path) for path in paths]
        self.simg.appendimages(paths)

        self.plotvariation()

    def rescan(self):
        self.cleartimeline()
        variationoperators.experiment = config.activeExperiment
        variation = self.simg.scan(self.operationindex)
        self.plotvariation(variation)

        for roi in self.viewbox.addedItems:
            # try:
            if hasattr(roi, 'isdeleting'):
                if not roi.isdeleting:
                    roi = roi.getArrayRegion(np.ones_like(self.imgview.imageItem.image), self.imageitem)
                    variation = self.simg.scan(self.operationindex, roi)
                    self.plotvariation(variation, [0, 255, 255])

                else:
                    self.viewbox.removeItem(roi)
                    # except Exception as ex:
                    # print 'Warning: error displaying ROI variation.'
                    #    print ex.message

    def setvariationmode(self, index):
        print 'operationset:', index
        self.operationindex = index

    def cleartimeline(self):
        for item in self.timeline.items:
            print item
            if type(item) is pg.PlotDataItem:
                item.isdeleting = True
                self.timeline.removeItem(item)

    def plotvariation(self, variation, color=None):
        if len(variation) == 0:
            return None

        if color is None:
            color = [255, 255, 255]

        # TODO: plot variation with indices, and skipped frames; skip None's



        t = np.array(variation.keys())
        d = np.array(variation.values())

        # print variation
        d = d[t.argsort()]
        t = sorted(t)

        self.timeline.enableAutoScale()
        # self.timeruler = TimeRuler(pen=pg.mkPen('#FFA500', width=3), movable=True)

        print 'plottype:', type(d[0])
        if type(d[0]) in [float, int, np.float64]:
            self.timeline.plot(t, d, pen=pg.mkPen(color=color))
        elif type(d[0]) is np.ndarray:
            for dt in d:
                self.timeline.plot(dt[0], dt[1])


    def redrawframe(self, index, time, forcelow=False):
        key = round(time)
        self.dimg = self.simg.getDiffImage(key)
        self.redrawimage(forcelow=forcelow)

    def gotomax(self):
        pass
        # self.parentwindow.timeruler.setValue(np.argmax(self.variationy))


class integrationwidget(QtGui.QTabWidget):
    sigReplot = QtCore.Signal()

    def __init__(self):
        super(integrationwidget, self).__init__()
        self.setTabPosition(self.West)
        self.qintegration = qintegrationwidget()
        self.chiintegration = chiintegrationwidget()
        self.addTab(self.qintegration, u'q')
        self.addTab(self.chiintegration, u'χ')
        self.currentChanged.connect(self.sigReplot)


class qintegrationwidget(pg.PlotWidget):
    def __init__(self):
        super(qintegrationwidget, self).__init__()
        self.setLabel('bottom', u'q (\u212B\u207B\u00B9)', '')
        self.qLine = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('#FFA500'))
        self.qLine.setVisible(False)
        self.addItem(self.qLine)


class chiintegrationwidget(pg.PlotWidget):
    def __init__(self):
        super(chiintegrationwidget, self).__init__()
        # self.chiintegration = chiintegrationwidget.getPlotItem()
        self.setLabel('bottom', u'χ (Degrees)')
        self.chiLine = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('#FFA500'))
        self.chiLine.setVisible(False)
        self.addItem(self.chiLine)


class ImageView(pg.ImageView):
    sigKeyRelease = QtCore.Signal()

    def buildMenu(self):
        super(ImageView, self).buildMenu()
        self.menu.removeAction(self.normAction)

    def keyReleaseEvent(self, ev):
        super(ImageView, self).keyReleaseEvent(ev)
        if ev.key() in [QtCore.Qt.Key_Right, QtCore.Qt.Key_Left, QtCore.Qt.Key_Up, QtCore.Qt.Key_Down]:
            ev.accept()
            self.sigKeyRelease.emit()


from scipy.signal import fftconvolve

class fxsviewer(ImageView):
    def __init__(self, paths=None, **kwargs):
        super(fxsviewer, self).__init__()
        if paths is None:
            paths = []

        conv = None
        avg = None
        for path in paths:
            dimg = loader.diffimage(filepath=path)
            if avg is None: avg = np.zeros((100, 100))

            def autocorrelate(A):
                return fftconvolve(A, A[::-1])

            conv = np.apply_along_axis(autocorrelate, axis=0, arr=dimg.cake)
            avg += conv[99:, :] + conv[:100, :]

        self.setImage(avg)





class pluginModeWidget(QtGui.QWidget):
    def __init__(self, plugins):
        super(pluginModeWidget, self).__init__()
        self.setLayout(QtGui.QHBoxLayout())

        font = QtGui.QFont()
        font.setPointSize(16)

        for key, plugin in plugins.items():
            button = QtGui.QPushButton(plugin.name)
            button.setFlat(True)
            button.setFont(font)
            button.isMode = True
            button.setAutoFillBackground(False)
            button.setCheckable(True)
            button.setAutoExclusive(True)
            button.clicked.connect(plugin.activate)
            if plugin is plugins.values()[0]:
                button.setChecked(True)
            self.layout().addWidget(button)
            if not plugin is plugins.values()[-1]:
                label = QtGui.QLabel('|')
                label.setFont(font)
                label.setStyleSheet('background-color:#111111;')
                self.layout().addWidget(label)


class previewwidget(pg.GraphicsLayoutWidget):
    """
    top-left preview
    """

    def __init__(self, tree):
        super(previewwidget, self).__init__()
        self.tree = tree
        self.model = tree.model()
        self.view = self.addViewBox(lockAspect=True)

        self.imageitem = pg.ImageItem()
        self.view.addItem(self.imageitem)
        self.imgdata = None
        self.setMinimumHeight(250)

        self.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)


    def loaditem(self, current, previous):

        try:
            path = self.model.filePath(current)
            if os.path.isfile(path):
                self.imgdata = loader.loadimage(path)
                self.imageitem.setImage(np.rot90(np.log(self.imgdata * (self.imgdata > 0) + (self.imgdata < 1)), 3),
                                        autoLevels=True)
        except TypeError:
            self.imageitem.clear()


class fileTreeWidget(QtGui.QTreeView):
    sigOpenFiles = QtCore.Signal(list)
    sigOpenDir = QtCore.Signal(str)

    def __init__(self):
        super(fileTreeWidget, self).__init__()
        self.filetreemodel = QtGui.QFileSystemModel()
        self.setModel(self.filetreemodel)
        self.filetreepath = pathtools.getRoot()
        self.treerefresh(self.filetreepath)
        header = self.header()
        self.setHeaderHidden(True)
        for i in range(1, 4):
            header.hideSection(i)
        filefilter = ['*' + ext for ext in
                      loader.acceptableexts]  # ["*.tif", "*.edf", "*.fits", "*.nxs", "*.hdf", "*.cbf"]
        self.filetreemodel.setNameFilters(filefilter)
        self.filetreemodel.setNameFilterDisables(False)
        self.filetreemodel.setResolveSymlinks(True)
        self.expandAll()
        self.sortByColumn(0, QtCore.Qt.AscendingOrder)
        self.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        self.setSelectionMode(self.ExtendedSelection)
        self.setIconSize(QtCore.QSize(16, 16))

        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.contextMenu)

        self.doubleClicked.connect(self.doubleclickevent)

    def contextMenu(self, position):
        menu = QtGui.QMenu()

        actionOpen = QtGui.QAction('Open', self)
        actionOpen.triggered.connect(self.openActionTriggered)
        menu.addAction(actionOpen)

        menu.exec_(self.viewport().mapToGlobal(position))

    def openActionTriggered(self):
        indices = self.selectedIndexes()
        paths = [self.filetreemodel.filePath(index) for index in indices]

        self.sigOpenFiles.emit(paths)

    def doubleclickevent(self, index):
        path = self.filetreemodel.filePath(index)

        if os.path.isfile(path):
            self.sigOpenFiles.emit([path])
        elif os.path.isdir(path):
            self.sigOpenDir.emit(path)
        else:
            print('Error on index (what is that?):', index)


    def treerefresh(self, path=None):
        """
        Refresh the file tree, or switch directories and refresh
        """
        if path is None:
            path = self.filetreepath

        root = QtCore.QDir(path)
        self.filetreemodel.setRootPath(root.absolutePath())
        self.setRootIndex(self.filetreemodel.index(root.absolutePath()))
        self.show()



def pixel2q(x, y, experiment):
    # SWITCH TO PYFAI GEOMETRY

    if x is None:
        x = experiment.getvalue('Center X')
    if y is None:
        y = experiment.getvalue('Center Y')

    r = np.sqrt((x - experiment.getvalue('Center X')) ** 2 + (y - experiment.getvalue('Center Y')) ** 2)
    theta = np.arctan2(r * experiment.getvalue('Pixel Size X'),
                       experiment.getvalue('Detector Distance'))
    wavelength = experiment.getvalue('Wavelength')
    return 4 * np.pi / wavelength * np.sin(theta / 2) * 1e-10