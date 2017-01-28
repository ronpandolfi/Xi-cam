# --coding: utf-8 --

from PySide import QtGui, QtCore
from PySide.QtCore import Qt
import numpy as np
import pyqtgraph as pg
from pipeline import loader, cosmics, integration, peakfinding, center_approx, variationoperators, pathtools
from xicam import config, ROI, debugtools, toolbar
from fabio import edfimage
import os
import pyqtgraph.parametertree.parameterTypes as pTypes
from pyqtgraph.parametertree import Parameter, ParameterTree, ParameterItem, registerParameterType
from xicam import dialogs
from xicam import xglobals
import scipy
from pipeline import variation
from xicam import threads
from pipeline import msg
from scipy.ndimage import morphology

class OOMTabItem(QtGui.QWidget):
    sigLoaded = QtCore.Signal()

    def __init__(self, itemclass=None, *args, **kwargs):
        """
        A collection of references that can be used to reconstruct an object dynamically and dispose of it when unneeded
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
        self.saved = None

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
                    msg.logMessage(self.kwargs['src'], msg.DEBUG)
                    imgdata = [loader.loadimage(path) for path in self.kwargs['src']]
                    imgdata = self.kwargs['operation'](imgdata)
                    dimg = loader.datadiffimage2(data=imgdata)
                    self.kwargs['dimg'] = dimg

            self.widget = self.itemclass(*self.args, **self.kwargs)

            if hasattr(self.widget,'restore'):
                self.widget.restore(*self.saved)

            self.layout().addWidget(self.widget)

            self.isloaded = True

            self.sigLoaded.emit()
            msg.logMessage(('Loaded:', self.itemclass),msg.DEBUG)

    def unload(self):
        """
        orphan the tab widget and queue them for deletion. Mwahahaha.
        Try to save values return from widget's save; restore values with widget's 'restore' method
        """

        if self.isloaded:
            if hasattr(self.widget,'save'):
                self.saved = self.widget.save()

            self.widget.deleteLater()
            self.widget = None
            self.isloaded = False


class dimgViewer(QtGui.QWidget):

    def __init__(self, dimg=None, src=None, plotwidget=None, toolbar=None, **kwargs):
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

        if dimg is not None:
            self.dimg = dimg
        else:
            self.dimg = loader.loaddiffimage(src)

        # Make an imageview for the image
        self.imgview = ImageView(self,actionLog_Intensity=self.toolbar.actionLog_Intensity)
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
        # menu.addAction(self.actionLogIntensity)
        # self.imgview.buildMenu()

        # Add a placeholder image item for the mask to the viewbox
        self.maskimage = pg.ImageItem(opacity=.25)
        self.viewbox.addItem(self.maskimage)

        # import ROI
        # self.arc=ROI.ArcROI((620.,29.),500.)
        # self.viewbox.addItem(self.arc)
        # print self.dimg.data
        # print self.imageitem


        # self.viewbox.addItem(pg.SpiralROI((0,0),1))

        try:
            energy = self.dimg.headers['Beamline Energy']
            if energy is not None:
                config.activeExperiment.setvalue('Energy', energy)
        except (AttributeError, TypeError, KeyError):
            msg.logMessage('Warning: Energy could not be determined from headers',msg.WARNING)

        # Force cache the detector
        _ = self.dimg.detector

        # Cache radial integration
        if self.dimg is not None:
            if self.dimg.rawdata is not None:
                self.redrawimage()

                # self.q, self.radialprofile = self.dimg.radialintegrate

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

        if xglobals.LUT is not None:
            hist = self.imgview.getHistogramWidget().item
            hist.setLevels(*xglobals.LUTlevels)
            hist.gradient.restoreState(xglobals.LUTstate)
            return True
        return False

    def resetLUT(self):

        msg.logMessage(('Levels:', self.imgview.getHistogramWidget().item.getLevels()),msg.DEBUG)
        # if self.imgview.getHistogramWidget().item.getLevels()==(0,1.):
        Lmax = np.nanmax(self.dimg.rawdata)

        self.imgview.autoLevels()
        msg.logMessage(('Levels set:', self.imgview.getHistogramWidget().item.getLevels()),msg.DEBUG)

    def cacheLUT(self):
        hist = self.imgview.getHistogramWidget().item
        xglobals.LUTlevels = hist.getLevels()
        xglobals.LUTstate = hist.gradient.saveState()
        xglobals.LUT = hist.getLookupTable(img=self.imageitem.image)

    # def send1Dintegration(self):
    # self.cache1Dintegration.emit(self.q, self.radialprofile)

    def removeROI(self, evt):

        # evt.scene().removeItem(evt)
        self.viewbox.removeItem(evt)
        self.replot()
        # evt.deleteLater()

        # self.viewbox.scene().removeItem(evt)

    # @property
    # def isChiPlot(self):
    #     return self.plotwidget.currentIndex()

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

                    data = self.dimg.transformdata

                    # if iscake:
                    # q = pixel2cake(x, y, self.dimg)
                    #
                    # elif isremesh:
                    # return
                    # else:
                    #     q = pixel2q(x, y, self.dimg.experiment)
                    # print x,y,self.dimg.data[int(x),int(y)],self.getq(x,y),self.getq(None,y),self.getq(x,None,),np.sqrt((x - self.dimg.experiment.center[0]) ** 2 + (y - self.dimg.experiment.center[1]) ** 2)
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
                    # y - self.dimg.experiment.center[1]) ** 2)))
                    # ,  r=%0.1f
                    if self.plotwidget is not None: #for timeline
                        self.plotwidget.movPosLine(self.getq(x, y),
                                                   self.getq(x, y, mode='parallel'),
                                                   self.getq(x, y, mode='z'))

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
                    self.plotwidget.qintegration.posLine.hide()

    def getq(self, x, y, mode=None):  # This is a mess...rewrite it sometime
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
            q = 4 * np.pi / wavelength * np.sin(theta / 2) * 1e-10
            if mode == 'parallel' and x < center[0]:
                return -q
            if mode == 'z' and y < center[1]:
                return -q
            return q

    def leaveEvent(self, evt):
        """
        hide crosshair and coordinates when mouse leaves viewer
        """
        self.hLine.setVisible(False)
        self.vLine.setVisible(False)
        # self.coordslabel.setVisible(False)
        if self.plotwidget:
            self.plotwidget.hidePosLine()

    def enterEvent(self, evt):
        """
        show crosshair and coordinates when mouse enters viewer
        """
        self.hLine.setVisible(True)
        self.vLine.setVisible(True)
        if hasattr(self.plotwidget, 'qintegration'):
            self.plotwidget.qintegration.posLine.setVisible(True)

    def redrawimage(self, returnimg=False):
        """
        redraws the diffraction image, checking drawing modes (log, symmetry, mask, cake)
        """

        self.dimg.cakemode = self.iscake
        self.dimg.remeshmode = self.isremesh
        self.dimg.radialsymmetrymode = self.isradialsymmetry
        self.dimg.mirrorsymmetrymode = self.ismirrorsymmetry
        self.dimg.logscale = self.islogintensity

        if returnimg:
            return self.dimg
        else:
            self.imgview.setImage(self.dimg)

        self.drawcenter()

        self.replot()

        if self.ismaskshown:
            self.maskoverlay()
        else:
            self.maskimage.clear()

        # if not iscake and not isremesh:
        #     self.imageitem.setRect(QtCore.QRect(0, 0, self.dimg.rawdata.shape[0], self.dimg.rawdata.shape[1]))

    def getcenter(self):
        if self.isremesh:
            remeshqpar = self.dimg.remeshqy
            remeshqz = self.dimg.remeshqx
            q = remeshqpar ** 2 + remeshqz ** 2
            center = np.where(q == q.min())
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
        xglobals.lastroi = (arc, self.imageitem)

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
            [self.getcenter()[0], -self.dimg.transformdata.shape[0]], 5, removable=True)
        region.sigRemoveRequested.connect(self.removeROI)
        self.viewbox.addItem(region)
        self.replot()
        region.sigRegionChangeFinished.connect(self.replot)
        xglobals.lastroi = (region, self.imageitem)
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
                                      bounds=[0, self.dimg.transformdata.shape[0]],
                                      values=[self.getcenter()[0] - 10,
                                              10 + self.getcenter()[0]])
        for line in region.lines:
            line.setPen(pg.mkPen('#00FFFF'))
        region.sigRegionChangeFinished.connect(self.replot)
        region.sigRemoveRequested.connect(self.removeROI)
        self.viewbox.addItem(region)
        self.replot()
        xglobals.lastroi = (region, self.imageitem)
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
                                      bounds=[0, self.dimg.transformdata.shape[1]],
                                      values=[self.getcenter()[1]-10,
                                              10 + self.getcenter()[1]])
        for line in region.lines:
            line.setPen(pg.mkPen('#00FFFF'))
        region.sigRegionChangeFinished.connect(self.replot)
        region.sigRemoveRequested.connect(self.removeROI)
        self.viewbox.addItem(region)

        self.replot()
        xglobals.lastroi = (region, self.imageitem)
        # else:
        # #self.viewbox.removeItem(self.region)
        #     self.region = None
        # self.replot()

    def removecosmics(self):
        c = cosmics.cosmicsimage(self.dimg.rawdata)
        c.run(maxiter=4)
        config.activeExperiment.addtomask(1-c.mask)
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
            # print item
            if issubclass(type(item), ROI.ArcROI):
                item.setPos(center)

    @debugtools.timeit
    def calibrate(self, algorithm, calibrant):

        algorithm(self.dimg, calibrant)

        self.replot()
        self.drawcenter()

    @debugtools.timeit
    def refinecenter(self):
        # Force cache the detector
        # _=self.dimg.detector

        cen = center_approx.refinecenter(self.dimg)
        config.activeExperiment.center = cen
        self.drawcenter()

    def getROIs(self):
        return [roi for roi in self.viewbox.addedItems if hasattr(roi, 'isdeleting') and not roi.isdeleting]

    def replot(self):
        self.plotwidget.replot()
        #widget(self.plotwidget.currentIndex()).replot(self.dimg,self.getROIs(),self.imageitem)
        # if self.plotwidget.currentIndex() == 0:
        #     self.plotwidget.qintegration.clear()
        #
        #     # self.replotprimary()
        #     self.plotwidget.qintegration.qLine = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('#FFA500'))
        #     self.plotwidget.qintegration.qLine.setVisible(False)
        #     self.plotwidget.qintegration.addItem(self.plotwidget.qintegration.qLine)
        #     self.replotq()
        # elif self.plotwidget.currentIndex() == 1:
        #     self.plotwidget.chiintegration.clear()
        #     self.plotwidget.chiintegration.chiLine = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('#FFA500'))
        #     self.plotwidget.chiintegration.chiLine.setVisible(False)
        #     self.plotwidget.chiintegration.addItem(self.plotwidget.chiintegration.chiLine)
        #     self.replotchi()

    # def replotchi(self):
    #     if not config.activeExperiment.iscalibrated:
    #         return None
    #
    #     cut = None
    #
    #
    #
    #     data = self.dimg.transformdata
    #
    #     ai = config.activeExperiment.getAI().getPyFAI()
    #     xglobals.pool.apply_async(integration.chiintegratepyFAI,
    #                               args=(self.dimg.data, self.dimg.mask, ai, self.iscake),
    #                               callback=self.chiintegrationrelay)
    #     # pipeline.integration.chiintegratepyFAI(self.dimg.data, self.dimg.mask, ai, precaked=self.iscake)
    #
    #     for roi in self.viewbox.addedItems:
    #         # try:
    #             if hasattr(roi, 'isdeleting'):
    #                 if not roi.isdeleting:
    #                     print type(roi)
    #                     cut = None
    #                     if issubclass(type(roi), pg.ROI) or issubclass(type(roi), pg.LinearRegionItem):
    #                         cut = (roi.getArrayRegion(np.ones_like(data), self.imageitem)).T
    #                         print 'Cut:', cut.shape
    #
    #
    #                     if cut is not None:
    #                         if self.iscake:
    #
    #                             ma = np.ma.masked_array(data, mask=1 - (cut * self.dimg.cakemask))
    #                             chi = self.dimg.cakeqy
    #                             I = np.ma.average(ma, axis=1)
    #                             I = np.trim_zeros(I, 'b')
    #                             chi = chi[:len(I)]
    #                             I = np.trim_zeros(I, 'f')
    #                             chi = chi[-len(I):]
    #                             self.plotchiintegration([chi, I, [0, 255, 255]])
    #                         else:
    #
    #                             ai = config.activeExperiment.getAI().getPyFAI()
    #                             self.pool.apply_async(integration.chiintegratepyFAI,
    #                                                   args=(self.dimg.transformdata, self.dimg.mask, ai, self.iscake, cut,
    #                                                         [0, 255, 255]),
    #                                                   callback=self.chiintegrationrelay)
    #
    #
    #
    #                 else:
    #                     self.viewbox.removeItem(roi)
    #
    # def replotq(self):
    #     if not config.activeExperiment.iscalibrated:
    #         return None
    #
    #     cut = None
    #
    #     data = self.dimg.transformdata
    #
    #     ai = config.activeExperiment.getAI().getPyFAI()
    #     print 'centeroverride:', [c * config.activeExperiment.getvalue('Pixel Size X') for c in
    #                               self.getcenter()[::-1]]
    #     print self.getcenter()
    #
    #     xglobals.pool.apply_async(integration.radialintegratepyFAI,
    #                               args=((self.dimg.transformdata),
    #                                     (self.dimg.mask if not self.isremesh else self.dimg.remeshmask), ai, None, None,
    #                                     [c * config.activeExperiment.getvalue('Pixel Size X') for c in
    #                                      self.getcenter()[::-1]]),
    #                               callback=self.qintegrationrelay)
    #
    #     for roi in self.viewbox.addedItems:
    #         # try:
    #             if hasattr(roi, 'isdeleting'):
    #                 if not roi.isdeleting:
    #                     print type(roi)
    #                     cut = None
    #                     if issubclass(type(roi), pg.ROI) or issubclass(type(roi), pg.LinearRegionItem):
    #
    #                         cut = (roi.getArrayRegion(np.ones_like(data), self.imageitem)).T
    #                         print 'Cut:', cut.shape
    #
    #
    #
    #
    #
    #
    #
    #                     elif type(roi) is pg.LineSegmentROI:
    #
    #                         cut = self.region.getArrayRegion(data, self.imageitem)
    #
    #                         x = np.linspace(self.viewbox.mapSceneToView(self.region.getSceneHandlePositions(0)[1]).x(),
    #                                         self.viewbox.mapSceneToView(self.region.getSceneHandlePositions(1)[1]).x(),
    #                                         cut.__len__())
    #                         y = np.linspace(self.viewbox.mapSceneToView(self.region.getSceneHandlePositions(0)[1]).y(),
    #                                         self.viewbox.mapSceneToView(self.region.getSceneHandlePositions(1)[1]).y(),
    #                                         cut.__len__())
    #
    #                         q = pixel2q(x, y, self.dimg.experiment)
    #                         qmiddle = q.argmin()
    #                         leftq = -q[0:qmiddle]
    #                         rightq = q[qmiddle:]
    #
    #                         if leftq.__len__() > 1: self.plotwidget.qintegration.plot(leftq, cut[:qmiddle])
    #                         if rightq.__len__() > 1: self.plotwidget.qintegration.plot(rightq, cut[qmiddle:])
    #
    #
    #                     if cut is not None:
    #                         if self.iscake:
    #
    #                             ma = np.ma.masked_array(data, mask=1 - (cut * self.dimg.cakemask))
    #                             q = self.dimg.cakeqx / 10.
    #                             I = np.ma.average(ma, axis=0)
    #                             I = np.trim_zeros(I, 'b')
    #                             q = q[:len(I)]
    #                             I = np.trim_zeros(I, 'f')
    #                             q = q[-len(I):]
    #                             self.plotqintegration([q, I, [0, 255, 255]])
    #                         else:
    #
    #                             ai = config.activeExperiment.getAI().getPyFAI()
    #                             xglobals.pool.apply_async(integration.radialintegratepyFAI, args=(
    #                                 (self.dimg.transformdata),
    #                                 (self.dimg.mask), ai, cut, #if not self.isremesh else self.dimg.remeshmask
    #                                 [0, 255, 255], [c * config.activeExperiment.getvalue('Pixel Size X') for c in
    #                                                 self.getcenter()[::-1]]),
    #                                                       callback=self.qintegrationrelay)
    #
    #
    #                 else:
    #                     self.viewbox.removeItem(roi)
    #         # except Exception as ex:
    #         #     print 'Warning: error displaying ROI integration.'
    #         #     print ex.message
    #
    # def qintegrationrelay(self, *args, **kwargs):
    #     self.sigPlotQIntegration.emit(*args, **kwargs)
    #
    # def plotqintegration(self, result):
    #     (q, radialprofile, color) = result
    #     if color is None:
    #         color = [255, 255, 255]
    #     # cyan:[0, 255, 255]
    #     curve = self.plotwidget.qintegration.plot(q, radialprofile, pen=pg.mkPen(color=color))
    #     curve.setZValue(3 * 255 - sum(color))
    #     self.plotwidget.qintegration.update()
    #
    # def chiintegrationrelay(self, *args, **kwargs):
    #     print args
    #     self.sigPlotChiIntegration.emit(*args, **kwargs)
    #
    # def plotchiintegration(self, result):
    #     (chi, chiprofile, color) = result
    #     if color is None:
    #         color = [255, 255, 255]
    #     # cyan:[0, 255, 255]
    #     curve = self.plotwidget.chiintegration.plot(chi, chiprofile, pen=pg.mkPen(color=color))
    #     curve.setZValue(3 * 255 - sum(color))
    #     self.plotwidget.chiintegration.update()

    def polymask(self):
        maskroi = None
        for roi in self.viewbox.addedItems:
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
                if 0 < p.y() < self.dimg.transformdata.shape[1] and 0 < p.x() < self.dimg.transformdata.shape[0]:
                    return True
                else:
                    return False

            self.maskROI.checkPointMove = checkPointMove

        else:  # If the mask is completed

            # Get the region of the image that was selected; unforunately the region is trimmed
            maskedarea = self.maskROI.getArrayRegion(np.ones_like(self.dimg.transformdata), self.imageitem,
                                                     returnMappedCoords=True)  # levels=(0, arr.max()
            # print maskedarea.shape

            # Decide how much to left and top pad based on the ROI bounding rectangle
            boundrect = self.viewbox.itemBoundingRect(self.maskROI)
            leftpad = max(boundrect.x(), 0)
            toppad = max(boundrect.y(), 0)


            # Pad the mask so it has the same shape as the image
            maskedarea = np.pad(maskedarea, ((int(leftpad), 0), (int(toppad), 0)), mode='constant')
            maskedarea = np.pad(maskedarea, (
                (0, self.dimg.transformdata.shape[0] - maskedarea.shape[0]), (0, self.dimg.transformdata.shape[1] - maskedarea.shape[1])),
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


    def thresholdmask(self):
        threshold, ok = QtGui.QInputDialog.getInt(self, 'Threshold value','Input intensity threshold:',3,0,10000000)
        print 'threshold:',threshold

        if ok and threshold:
            mask = self.dimg.rawdata>threshold

            morphology.binary_closing(mask,morphology.generate_binary_structure(2,2),output=mask) # write-back to mask

            config.activeExperiment.addtomask(mask)


    def maskoverlay(self):
        if self.iscake:
            mask = self.dimg.cakemask
        elif self.isremesh:
            mask = self.dimg.remeshmask
        else:
            mask = config.activeExperiment.mask

        if mask is None:
            self.maskimage.clear()
        else:
            # Draw the mask as a red channel image with an alpha mask
            msg.logMessage(('maskmax:', np.max(mask)),msg.DEBUG)
            invmask = 1 - mask
            self.maskimage.setImage(
                np.dstack((invmask, np.zeros_like(invmask), np.zeros_like(invmask), invmask)).astype(np.float),
                opacity=.5)
            self.maskimage.setLevels([0,1])

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
        data = self.imageitem.image
        guesspath = self.paths[0]
        dialogs.savedatadialog(data=data, guesspath=guesspath, headers=self.dimg.headers)

    def capture(self):
        captureroi = None
        for roi in self.viewbox.addedItems:
            if type(roi) is pg.RectROI:
                captureroi = roi

        if captureroi is None:

            # Add ROI item to the image
            self.captureROI = pg.RectROI(config.activeExperiment.center, (100, 100))
            self.viewbox.addItem(self.captureROI)

            # Override the ROI's function to check if any points will be moved outside the boundary; False prevents move
            def checkPointMove(handle, pos, modifiers):
                p = self.viewbox.mapToView(pos)
                if 0 < p.y() < self.dimg.data.shape[0] and 0 < p.x() < self.dimg.data.shape[1]:
                    return True
                else:
                    return False

            self.captureROI.checkPointMove = checkPointMove

        else:  # If the mask is completed

            # Get the shape
            lowerleft = [max(int(c), 0) for c in self.captureROI.pos()]
            topright = [max(int(s + p), 0) for s, p in zip(self.captureROI.size(), self.captureROI.pos())]

            dataregion = self.dimg.rawdata[lowerleft[0]:topright[0], lowerleft[1]:topright[1]]
            maskregion = self.dimg.mask[lowerleft[0]:topright[0], lowerleft[1]:topright[1]]
            guesspath = self.filepaths[0]

            qpar_min = self.getq(*lowerleft, mode='parallel') * 10
            qvrt_min = self.getq(*lowerleft, mode='z') * 10
            qpar_max = self.getq(*topright, mode='parallel') * 10
            qvrt_max = self.getq(*topright, mode='z') * 10

            headers = {'qpar_min': qpar_min, 'qpar_max': qpar_max, 'qvrt_min': qvrt_min, 'qvrt_max': qvrt_max}
            dialogs.savedatadialog(data=dataregion, mask=maskregion, headers=headers, guesspath=guesspath)

            # Remove the ROI
            self.viewbox.removeItem(self.captureROI)

    def clearsgoverlays(self):
        for item in self.viewbox.addedItems:
            from pipeline import spacegroups

            # print type(item), type(item) is spacegroups.peakoverlay
            if type(item) is spacegroups.peakoverlay:
                self.viewbox.removeItem(item)  # item is marked for removal, but must be hidden removal takes effect
                item.hide()

    def drawsgoverlay(self, peakoverlay):
        self.clearsgoverlays()

        peakoverlay.setCenter(*self.getcenter())

        #peakoverlay.setRect(QtCore.QRect(self.getcenter()[0],self.getcenter()[1],1,1))
        self.viewbox.addItem(peakoverlay)
        peakoverlay.enable(self.viewbox)


class timelineViewer(dimgViewer):
    sigAddTimelineData = QtCore.Signal(tuple, list)
    sigClearTimeline = QtCore.Signal()
    def __init__(self, simg=None, files=None, toolbar=None):
        self.variationcurve = dict()
        self.toolbar = toolbar

        if simg is None:
            simg = loader.loaddiffimage(files)

        self.simg = simg
        #dimg = simg.first()

        self.operationindex = 0

        super(timelineViewer, self).__init__(simg, toolbar=toolbar)

        #img = np.array(self.simg.thumbs)
        #img = (np.log(img * (img > 0) + (img < 1)))

        self.imgview.setImage(simg, xvals=simg.xvals(''))

        # self.imageitem.sigImageChanged.connect(self.setscale)

        # self.paths = dict(zip(range(len(paths)), sorted(paths)))
        # self.setvariationmode(0)
        # self.gotomax()
        # self.imgview.timeLine.sigPositionChanged.disconnect(self.imgview.timeLineChanged)

        self.highresimgitem = pg.ImageItem()

        self.viewbox.addItem(self.highresimgitem)
        self.highresimgitem.hide()

        # self.imgview.timeLine.sigPositionChangeFinished.connect(self.drawframeoverlay)
        # self.imgview.timeLine.sigDragged.connect(self.hideoverlay)
        # self.imgview.sigKeyRelease.connect(self.drawframeoverlay)

        # self.drawframeoverlay()
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
        # need to connect it
        menu.addAction(opwidgetaction)

        # self.imgview.getHistogramWidget().item.setImageItem(self.highresimgitem)
        # self.imgview.getHistogramWidget().item.sigLevelChangeFinished.connect(self.updatelowresLUT)


        self.rescan()

    def redrawimage(self, returnimg=False):
        self.simg.cakemode = self.iscake
        self.simg.remeshmode = self.isremesh
        self.simg.radialsymmetrymode = self.isradialsymmetry
        self.simg.mirrorsymmetrymode = self.ismirrorsymmetry
        self.simg.logscale = self.islogintensity
        super(timelineViewer, self).redrawimage(returnimg)
        timelineplot = self.imgview.getRoiPlot()
        self.timeline = timelineplot.getPlotItem()
        self.timeline.getViewBox().setMouseEnabled(x=False, y=True)

    def processtimeline(self):
        self.toolbar.actionProcess.setChecked(False)
        self.rescan()

    def aborttimeline(self):
        pass

    def reduce(self):
        pass
        # self.skipframes = (self.variationy[0:-1] / self.variationy[1:]) > 0.1

    def appendimage(self, d, paths):  # WIP
        paths = [os.path.join(d, path) for path in paths]
        self.simg.appendimages(paths)



    def rescan(self):
        #return
        self.cleartimeline()
        variationoperators.experiment = config.activeExperiment


        # keys = range(len(self.simg))[:-1]# TODO: generalize!
        # values = list(variation.variationiterator(self.simg))[1:]
        # print 'keys:',keys
        # print 'values:',values
        # d = dict(zip(keys,values))
        # self.plotvariation(d)

        # Run on thread queue
        bg_variation = threads.iterator(callback_slot=self.sigAddTimelineData,
                                        finished_slot=self.processingfinished,
                                        parent=self)(variation.variationiterator)
        bg_variation(self.simg, self.operationindex)
        # xglobals.pool.apply_async(variation.scanvariation,args=(self.simg.filepaths),callback=self.testreceive)

        # variation = self.simg.scan(self.operationindex)
        # self.plotvariation(variation)

        for roi in self.viewbox.addedItems:
            # try:
            if hasattr(roi, 'isdeleting'):
                if not roi.isdeleting:
                    roi = roi.getArrayRegion(np.ones_like(self.imgview.imageItem.image), self.imageitem).T
                    # variation = self.simg.scan(self.operationindex, roi)
                    # self.plotvariation(variation, [0, 255, 255])
                    # Run on thread queue
                    # TODO: pull color from ROI, give ROIs deterministic colors with pyqtgraph.intColor
                    bg_variation(self.simg, self.operationindex, roi=roi, color=[0, 255, 255])
                else:
                    self.viewbox.removeItem(roi)
                    # except Exception as ex:
                    # print 'Warning: error displaying ROI variation.'
                    #    print ex.message


    def processingfinished(self, *args, **kwargs):
        msg.showMessage('Processing complete.',4)

    def setvariationmode(self, index):
        self.operationindex = index

    def cleartimeline(self):
        self.sigClearTimeline.emit()

    # def plotvariation(self, variation, color=None):
    #     if len(variation) == 0:
    #         return None
    #
    #     if color is None:
    #         color = [255, 255, 255]
    #
    #     # TODO: plot variation with indices, and skipped frames; skip None's
    #
    #
    #
    #     t = np.array(variation.keys())
    #     d = np.array(variation.values())
    #
    #     # print variation
    #     d = d[t.argsort()]
    #     t = sorted(t)
    #
    #     self.timeline.enableAutoScale()
    #     # self.timeruler = TimeRuler(pen=pg.mkPen('#FFA500', width=3), movable=True)
    #
    #     print 'plottype:', type(d[0])
    #     if type(d[0]) in [float, int, np.float64]:
    #         self.timeline.plot(t, d, pen=pg.mkPen(color=color))
    #     elif type(d[0]) is np.ndarray:
    #         for dt in d:
    #             self.timeline.plot(dt[0], dt[1])

    # def redrawframe(self, index, time, forcelow=False):
    #     key = round(time)
    #     self.dimg = self.simg.getDiffImage(key)
    #     self.redrawimage(forcelow=forcelow)

    def gotomax(self):
        pass
        # self.parentwindow.timeruler.setValue(np.argmax(self.variationy))

    def replot(self):
        pass


class integrationwidget(QtGui.QTabWidget):

    def __init__(self, getViewer):
        super(integrationwidget, self).__init__()
        self.setTabPosition(self.West)

        self.getViewer = getViewer

        self.qintegration = qintegrationwidget()
        self.chiintegration = chiintegrationwidget()
        self.xintegration = xintegrationwidget()
        self.zintegration = zintegrationwidget()

        self.cakexintegration = cakexintegrationwidget()
        self.cakezintegration = cakezintegrationwidget()

        self.remeshqintegration = remeshqintegrationwidget()
        self.remeshchiintegration = remeshchiintegrationwidget()
        self.remeshxintegration = remeshxintegrationwidget()
        self.remeshzintegration = remeshzintegrationwidget()

        self.currentChanged.connect(self.replot)


    def movPosLine(self,*args,**kwargs):
        if self.currentIndex() == -1: return
        self.widget(self.currentIndex()).movPosLine(*args,**kwargs)

    def updatemodes(self):
        if not self.getViewer():
            return

        previouswidget = self.currentWidget()

        self.blockSignals(True)

        for i in range(self.count(),-1,-1):
            self.removeTab(i)

        if self.getViewer().iscake:
            self.addTab(self.cakexintegration, u'x')
            self.addTab(self.cakezintegration, u'z')
        elif self.getViewer().isremesh:
            self.addTab(self.remeshqintegration, u'q')
            self.addTab(self.remeshchiintegration, u'χ')
            self.addTab(self.remeshxintegration, u'x')
            self.addTab(self.remeshzintegration, u'z')
        else:
            self.addTab(self.qintegration, u'q')
            self.addTab(self.chiintegration, u'χ')
            self.addTab(self.xintegration, u'x')
            self.addTab(self.zintegration, u'z')

        newwidgets = [self.widget(i) for i in range(self.count())]
        if previouswidget in newwidgets: self.setCurrentWidget(previouswidget)

        self.blockSignals(False)



    def replot(self):
        viewer = self.getViewer()
        if not viewer: return
        self.updatemodes()
        dimg = viewer.dimg
        rois = viewer.getROIs()
        imageitem = viewer.imageitem
        self.widget(self.currentIndex()).replot(dimg,rois,imageitem)

    def hidePosLine(self):
        if self.currentIndex() > -1:
            self.widget(self.currentIndex()).posLine.hide()


class integrationsubwidget(pg.PlotWidget):
    integrationfunction = None
    iscake = False
    isremesh = False

    def __init__(self,axislabel):
        super(integrationsubwidget, self).__init__()
        self.setLabel('bottom', axislabel, '')
        self.posLine = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('#FFA500'))
        self.posLine.setVisible(False)
        self.addItem(self.posLine)

        self.iscleared = True
        self.requestkey = 0


    def replot(self, dimg, rois, imageitem):
        data = dimg.transformdata
        mask = dimg.transformmask
        if self.integrationfunction is None:
            raise NotImplementedError
        try:
            self.applyintegration(self.integrationfunction,dimg,rois,data,mask,imageitem)
        except ValueError:
            msg.logMessage('Maybe the roi was too far away?',msg.DEBUG)

    def replotcallback(self,*args,**kwargs):
        self.sigPlotResult.emit(*args, **kwargs)

    def applyintegration(self,integrationfunction,dimg,rois,data,mask,imageitem):
        self.requestkey += 1
        self.iscleared = False

        if self.iscake:
            qvrt = dimg.cakeqy
            qpar = dimg.cakeqx
        elif self.isremesh:
            qvrt = dimg.remeshqy
            qpar = dimg.remeshqx
        else:
            qvrt = None
            qpar = None

        # xglobals.pool.apply_async(integrationfunction, args=(data, mask, dimg.experiment.getAI().getPyFAI(), None,
        #                                                                None, self.requestkey, qvrt, qpar),
        #                           callback=self.replotcallback)

        runnable = threads.RunnableMethod(integrationfunction, method_args=(
        data, mask, dimg.experiment.getAI().getPyFAI(), None, None, self.requestkey, qvrt, qpar),
                                          callback_slot=self.replotcallback)
        threads.add_to_queue(runnable)
        # replot roi integration
        for roi in rois:
            if roi.isdeleting:
                roi.deleteLater()
                continue

            cut = (roi.getArrayRegion(np.ones_like(dimg.transformdata), imageitem)).T
            msg.logMessage(('Cut:', cut.shape),msg.DEBUG)

            if cut is not None:
                # xglobals.pool.apply_async(integrationfunction,
                #                           args=(data, mask, dimg.experiment.getAI().getPyFAI(), cut, [0, 255, 255], self.requestkey, qvrt, qpar),
                #                           callback=self.replotcallback)
                runnable = threads.RunnableMethod(integrationfunction, method_args=(
                data, mask, dimg.experiment.getAI().getPyFAI(), cut, [0, 255, 255], self.requestkey, qvrt, qpar),
                                                  callback_slot=self.replotcallback)
                threads.add_to_queue(runnable)
    def movPosLine(self, q,qx,qz,dimg=None):
        pass #raise NotImplementedError

    def plotresult(self, result):

        (x, y, color,requestkey) = result
        if requestkey == self.requestkey:
            if not self.iscleared:
                self.plotItem.clear()
                self.addItem(self.posLine)
                self.iscleared = True
            if color is None:
                color = [255, 255, 255]
            y[y<=0]=1.E-9
            curve = self.plotItem.plot(x, y, pen=pg.mkPen(color=color))
            curve.setZValue(3 * 255 - sum(color))

            self.plotItem.update()


class qintegrationwidget(integrationsubwidget):

    sigPlotResult = QtCore.Signal(object)
    integrationfunction = staticmethod(integration.qintegrate)

    def __init__(self):
        super(qintegrationwidget, self).__init__(axislabel=u'q (\u212B\u207B\u00B9)')
        self.sigPlotResult.connect(self.plotresult)

    def movPosLine(self,q,qx,qz,dimg=None):
        self.posLine.setPos(q)
        self.posLine.show()


class chiintegrationwidget(integrationsubwidget):

    sigPlotResult = QtCore.Signal(object)
    integrationfunction = staticmethod(integration.chiintegratepyFAI)

    def __init__(self):
        super(chiintegrationwidget, self).__init__(axislabel=u'χ (Degrees)')
        self.sigPlotResult.connect(self.plotresult)

    def movPosLine(self,q, qx, qz, dimg=None):
        self.posLine.setPos(np.rad2deg(np.arctan2(qz,qx)))
        self.posLine.show()

class xintegrationwidget(integrationsubwidget):

    sigPlotResult = QtCore.Signal(object)
    integrationfunction = staticmethod(integration.xintegrate)

    def __init__(self):
        super(xintegrationwidget, self).__init__(axislabel=u'q<sub>x</sub> (\u212B\u207B\u00B9)')
        self.sigPlotResult.connect(self.plotresult)

    def movPosLine(self,q, qx, qz, dimg=None):
        self.posLine.setPos(qx)
        self.posLine.show()

class zintegrationwidget(integrationsubwidget):

    sigPlotResult = QtCore.Signal(object)
    integrationfunction = staticmethod(integration.zintegrate)

    def __init__(self):
        super(zintegrationwidget, self).__init__(axislabel=u'q<sub>z</sub> (\u212B\u207B\u00B9)')
        self.sigPlotResult.connect(self.plotresult)

    def movPosLine(self,q, qx, qz, dimg=None):
        self.posLine.setPos(qz)
        self.posLine.show()

class cakexintegrationwidget(integrationsubwidget):

    iscake = True
    sigPlotResult = QtCore.Signal(object)
    integrationfunction = staticmethod(integration.cakexintegrate)

    def __init__(self):
        super(cakexintegrationwidget, self).__init__(axislabel=u'χ (Degrees)')
        self.sigPlotResult.connect(self.plotresult)

    def movPosLine(self,q, qx, qz, dimg=None):
        self.posLine.setPos(np.rad2deg(np.arctan2(qx,qz)))
        self.posLine.show()

class cakezintegrationwidget(integrationsubwidget):

    iscake = True
    sigPlotResult = QtCore.Signal(object)
    integrationfunction = staticmethod(integration.cakezintegrate)

    def __init__(self):
        super(cakezintegrationwidget, self).__init__(axislabel=u'q (\u212B\u207B\u00B9)')
        self.sigPlotResult.connect(self.plotresult)

    def movPosLine(self,q, qx, qz, dimg=None):
        self.posLine.setPos(q)
        self.posLine.show()

class remeshqintegrationwidget(integrationsubwidget):

    isremesh=True
    sigPlotResult = QtCore.Signal(object)
    integrationfunction = staticmethod(integration.remeshqintegrate)

    def __init__(self):
        super(remeshqintegrationwidget, self).__init__(axislabel=u'q (\u212B\u207B\u00B9)')
        self.sigPlotResult.connect(self.plotresult)

    def movPosLine(self,q,qx,qz,dimg=None):
        self.posLine.setPos(q)
        self.posLine.show()

class remeshchiintegrationwidget(integrationsubwidget):

    isremesh=True
    sigPlotResult = QtCore.Signal(object)
    integrationfunction = staticmethod(integration.remeshchiintegrate)

    def __init__(self):
        super(remeshchiintegrationwidget, self).__init__(axislabel=u'χ (Degrees)')
        self.sigPlotResult.connect(self.plotresult)

    def movPosLine(self,q, qx, qz, dimg=None):
        self.posLine.setPos(np.rad2deg(np.arctan2(qz, qx)))
        self.posLine.show()

class remeshxintegrationwidget(integrationsubwidget):

    isremesh=True
    sigPlotResult = QtCore.Signal(object)
    integrationfunction = staticmethod(integration.remeshxintegrate)

    def __init__(self):
        super(remeshxintegrationwidget, self).__init__(axislabel=u'q (\u212B\u207B\u00B9)')
        self.sigPlotResult.connect(self.plotresult)

    def movPosLine(self,q, qx,qz,dimg=None):
        self.posLine.setPos(qx)
        self.posLine.show()

class remeshzintegrationwidget(integrationsubwidget):

    isremesh=True
    sigPlotResult = QtCore.Signal(object)
    integrationfunction = staticmethod(integration.remeshzintegrate)

    def __init__(self):
        super(remeshzintegrationwidget, self).__init__(axislabel=u'q (\u212B\u207B\u00B9)')
        self.sigPlotResult.connect(self.plotresult)

    def movPosLine(self,q, qx,qz,dimg=None):
        self.posLine.setPos(qz)
        self.posLine.show()


def getHistogram(self, bins='auto', step='auto', targetImageSize=None, targetHistogramSize=500, **kwds):
    """Returns x and y arrays containing the histogram values for the current image.
    For an explanation of the return format, see numpy.histogram().

    The *step* argument causes pixels to be skipped when computing the histogram to save time.
    If *step* is 'auto', then a step is chosen such that the analyzed data has
    dimensions roughly *targetImageSize* for each axis.

    The *bins* argument and any extra keyword arguments are passed to
    np.histogram(). If *bins* is 'auto', then a bin number is automatically
    chosen based on the image characteristics:

    * Integer images will have approximately *targetHistogramSize* bins,
      with each bin having an integer width.
    * All other types will have *targetHistogramSize* bins.

    This method is also used when automatically computing levels.
    """
    if self.image is None:
        return None, None

    if not targetImageSize: targetImageSize = min(200, self.image.shape[0], self.image.shape[1])

    if step == 'auto':
        step = (np.ceil(self.image.shape[0] / targetImageSize),
                np.ceil(self.image.shape[1] / targetImageSize))
    if np.isscalar(step):
        step = (step, step)
    stepData = self.image[::int(step[0]), ::int(step[1])]

    if bins == 'auto':
        if stepData.dtype.kind in "ui":
            mn = stepData.min()
            mx = stepData.max()
            step = np.ceil((mx - mn) / 500.)
            bins = np.arange(mn, mx + 1.01 * step, step, dtype=np.int)
            if len(bins) == 0:
                bins = [mn, mx]
        else:
            bins = 500

    kwds['bins'] = bins
    hist = np.histogram(stepData, **kwds)
    hist[0][0] = 0
    return hist[1][:-1], hist[0]


pg.ImageItem.getHistogram = getHistogram

class ImageView(pg.ImageView):
    sigKeyRelease = QtCore.Signal()
    def __init__(self,*args,**kwargs):
        self.actionLog_Intensity=kwargs['actionLog_Intensity']
        del kwargs['actionLog_Intensity']
        super(ImageView, self).__init__(*args,**kwargs)

    def buildMenu(self):
        super(ImageView, self).buildMenu()
        self.menu.removeAction(self.normAction)

    def timeIndex(self, slider):
        ## Return the time and frame index indicated by a slider
        if self.image is None:
            return (0, 0)

        t = slider.value()

        xv = self.tVals
        if xv is None:
            ind = int(t)
        else:
            if len(xv) < 2:
                return (0, 0)
            inds = np.argwhere(xv <= t)  # <- The = is import to reach the last value
            if len(inds) < 1:
                return (0, t)
            ind = inds[-1, 0]
        return ind, t

    def setImage(self,*args,**kwargs):
        super(ImageView, self).setImage(*args,**kwargs)
        # if self.actionLog_Intensity.isChecked():
        #     levelmin = np.log(self.levelMin)
        #     levelmax = np.log(self.levelMax)
        #     if np.isnan(levelmin): levelmin = 0
        #     if np.isnan(levelmax): levelmax = 1
        #     if np.isinf(levelmin): levelmin = 0
        #     msg.logMessage(('min:',levelmin),msg.DEBUG)
        #     msg.logMessage(('max:',levelmax),msg.DEBUG)
        #
        #     self.ui.histogram.setLevels(levelmin, levelmax)

    # def updateImage(self, autoHistogramRange=True): # inject logarithm action
    #     ## Redraw image on screen
    #     if self.image is None:
    #         return
    #
    #     image = self.getProcessedImage()
    #
    #     if autoHistogramRange:
    #         self.ui.histogram.setHistogramRange(self.levelMin, self.levelMax)
    #     if self.axes['t'] is None:
    #         self.imageItem.updateImage(np.log(image * (image> 0) + (image < 1)) if self.actionLog_Intensity.isChecked() else image)
    #     else:
    #         self.ui.roiPlot.show()
    #         self.imageItem.updateImage(np.log(image[self.currentIndex] * (image[self.currentIndex]> 0) + (image[self.currentIndex] < 1)) if self.actionLog_Intensity.isChecked() else image[self.currentIndex])


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


class xasviewer(pg.PlotWidget):
    def __init__(self, paths, *args, **kwargs):
        super(xasviewer, self).__init__(*args, **kwargs)
        for path in paths:
            for i in range(2, 16):
                self.plot(loader.loadxfs(path)[:, 2], loader.loadxfs(path)[:, i], pen=(i, 16))


class pluginModeWidget(QtGui.QWidget):
    def __init__(self, plugins):
        super(pluginModeWidget, self).__init__()
        self.setLayout(QtGui.QHBoxLayout())

        self.font = QtGui.QFont()
        self.font.setPointSize(16)
        self.plugins = plugins

        self.reload()

    def reload(self):
        w = self.layout().takeAt(0)
        while w:
            w.widget().deleteLater()
            del w
            w = self.layout().takeAt(0)

        for key, plugin in self.plugins.items():
            if plugin.enabled:
                if plugin.instance.hidden:
                    continue

                button = QtGui.QPushButton(plugin.name)
                button.setFlat(True)
                button.setFont(self.font)
                button.setProperty('isMode', True)
                button.setAutoFillBackground(False)
                button.setCheckable(True)
                button.setAutoExclusive(True)
                button.clicked.connect(plugin.activate)
                if plugin is self.plugins.values()[0]:
                    button.setChecked(True)
                self.layout().addWidget(button)
                label = QtGui.QLabel('|')
                label.setFont(self.font)
                label.setStyleSheet('background-color:#111111;')
                self.layout().addWidget(label)

        self.layout().takeAt(self.layout().count() - 1).widget().deleteLater()  # Delete the last pipe symbol


class previewwidget(pg.GraphicsLayoutWidget):
    """
    top-left preview
    """

    def __init__(self):
        super(previewwidget, self).__init__()
        self.view = self.addViewBox(lockAspect=True, enableMenu=False)
        self.imageitem = pg.ImageItem()
        self.textitem = pg.TextItem()
        self.imgdata = None
        self.setMinimumHeight(250)

        self.view.addItem(self.imageitem)
        self.view.addItem(self.textitem)
        self.textitem.hide()
        self.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)

        # def textItemBounds(axis, frac=1.0, orthoRange=None):
        #     b = self.textitem.boundingRect()
        #     sx, sy = self.view.viewPixelSize()
        #     x, y = sx*b.width(), sy*b.height()
        #     if axis == 0: return (-x/2, x/2)
        #     if axis == 1: return (0, y)
        #
        # self.textitem.dataBounds = textItemBounds

    def loaditem(self, item):
        if isinstance(item, str) or isinstance(item, unicode):
            if os.path.isfile(item) or os.path.isdir(item):
                item = loader.loadimage(item)
            else:
                self.setText(item)
                return
        try:
            self.setImage(item)
        except TypeError:
            self.imageitem.clear()

    def setImage(self, imgdata):
        self.imageitem.clear()
        self.textitem.hide()
        self.imgdata = imgdata
        self.imageitem.setImage(np.rot90(np.log(self.imgdata * (self.imgdata > 0) + (self.imgdata < 1)), 3),
                                autoLevels=True)

    def setText(self, text):
        self.textitem.setText(text)
        self.textitem.setFont(QtGui.QFont('Zero Threes'))  # Not working for some reason
        self.imageitem.clear()
        self.textitem.show()


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
            msg.logMessage(('Error on index (what is that?):', index),msg.ERROR)

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


class frameproptable(pg.TableWidget):
    """
    Widget for displaying hierarchical python data structures
    (eg, nested dicts, lists, and arrays), adapted from pyqtgraph datatree.
    """

    def __init__(self, parent=None, data=None):
        super(frameproptable, self).__init__(parent)
        self.setVerticalScrollMode(self.ScrollPerPixel)
        self.setData(data)
        self.setColumnCount(2)
        self.setHorizontalHeaderLabels(['Parameter', 'Value'])
        self.horizontalHeader().setStretchLastSection(True)
        self.verticalHeader().hide()
        self.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding,QtGui.QSizePolicy.MinimumExpanding)
        self.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
        self.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)

        useAsMenu = QtGui.QMenu(u'Use as...',parent=self.contextMenu)
        useAsMenu.addAction('Beam Energy').triggered.connect(self.useAsEnergy)
        useAsMenu.addAction('Downstream Intensity').triggered.connect(self.useAsI1)
        useAsMenu.addAction('Timeline Axis').triggered.connect(self.useAsTimeline)
        self.contextMenu.addMenu(useAsMenu)

    def setData(self,data):
        if data is None:
            self.setVisible(False)
            return
        data=sorted(data.items())
        self.setHidden(len(data) == 0)
        super(frameproptable, self).setData(data)

    def sizeHint(self):
        return QtCore.QSize(self.parent().width(),self.parent().height()/2)

    def useAsI1(self):
        config.activeExperiment.setHeaderMap('I1 AI',self.getSelectedKey())

    def useAsEnergy(self):
        config.activeExperiment.setHeaderMap('Beam Energy',self.getSelectedKey())

    def useAsTimeline(self):
        config.activeExperiment.setHeaderMap('Timeline Axis', self.getSelectedKey())

    def getSelectedKey(self):
        return self.item(self.selectedIndexes()[0].row(),0).value


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


class filesListWidget(QtGui.QWidget):
    def __init__(self, *args, **kwargs):
        super(filesListWidget, self).__init__()
        self.horizontalLayout = QtGui.QHBoxLayout(self)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.paths = QtGui.QListWidget(self)
        self.paths.setSelectionMode(QtGui.QAbstractItemView.MultiSelection)
        self.paths.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
        self.paths.setObjectName("paths")
        self.horizontalLayout.addWidget(self.paths)
        self.verticalLayout_8 = QtGui.QVBoxLayout()
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.addfilesbutton = QtGui.QToolButton(self)
        self.addfilesbutton.setObjectName("addfiles")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("xicam/gui/icons_43.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.addfilesbutton.setIcon(icon)
        self.verticalLayout_8.addWidget(self.addfilesbutton)
        self.removefilesbutton = QtGui.QToolButton(self)
        self.removefilesbutton.setObjectName("removefiles")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("xicam/gui/icons_58.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.removefilesbutton.setIcon(icon2)
        self.verticalLayout_8.addWidget(self.removefilesbutton)
        spacerItem = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.verticalLayout_8.addItem(spacerItem)
        self.horizontalLayout.addLayout(self.verticalLayout_8)
        self.addfilesbutton.clicked.connect(self.addfiles)
        self.removefilesbutton.clicked.connect(self.removefiles)

    def addfiles(self):
        paths, ok = QtGui.QFileDialog.getOpenFileNames(None, 'Add files to Batch', os.curdir,
                                                       '*' + ' *'.join(loader.acceptableexts))
        self.paths.addItems(paths)

    def removefiles(self):
        for index in self.paths.selectedIndexes():
            item = self.paths.takeItem(index.row())
            item = None
