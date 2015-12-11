import pyqtgraph as pg
from PySide import QtGui, QtCore
import viewer
import numpy as np
import pipeline
import os
import debugtools


class timelinetabtracker(QtGui.QWidget):
    """
    A light-weight version of the timeline tab that is retained when the full tab is disposed. Used to generate tab.
    """
    def __init__(self, paths, experiment, parent):
        super(timelinetabtracker, self).__init__()

        self.paths = paths
        self.experiment = experiment
        self.parent = parent
        self.tab = None

        parent.listmodel.widgetchanged()

        self.isloaded = False

    def load(self):
        if not self.isloaded:
            self.layout = QtGui.QHBoxLayout(self)
            simg = pipeline.loader.imageseries(self.paths, self.experiment)
            self.tab = timelinetab(simg, self.parent)
            self.layout.addWidget(self.tab)

            # print('Successful load! :P')
            self.isloaded = True


    def unload(self):
        if self.isloaded:
            self.layout.parent = None
            self.layout.deleteLater()
            # self.tab = None
            print('Successful unload!')
            self.isloaded = False


class timelinetab(viewer.imageTab):
    def __init__(self, simg, parentwindow):
        self.variation = dict()
        self.simg = simg
        dimg = simg.first()

        self.operationindex=0


        super(timelinetab, self).__init__(dimg, parentwindow)

        img = np.array(self.simg.thumbs)
        img = (np.log(img * (img > 0) + (img < 1)))

        self.imgview.setImage(img, xvals=self.simg.xvals)

        self.imageitem.sigImageChanged.connect(self.setscale)

        # self.paths = dict(zip(range(len(paths)), sorted(paths)))
        self.parentwindow = parentwindow
        #self.setvariationmode(0)
        #self.gotomax()
        self.istimeline = True
        #self.imgview.timeLine.sigPositionChanged.disconnect(self.imgview.timeLineChanged)

        self.highresimgitem=pg.ImageItem()

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
        operationcombo.addItems(pipeline.variationoperators.operations.keys())
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
        self.parentwindow.timelinetoolbar.actionProcess.setChecked(False)

    def aborttimeline(self):
        pass

    def plottimeline(self, t, V, color=[255, 255, 255]):
        pass

    def drawframeoverlay(self):
        self.scale=1
        self.dimg = self.simg.getDiffImage(round(self.imgview.timeLine.getXPos()))
        self.imgview.imageItem.updateImage(self.redrawimage(returnimg=True),noscale=True)


    def updatelowresLUT(self):

        self.imageitem.setLookupTable(self.imgview.getHistogramWidget().item.getLookupTable)


    def hideoverlay(self):
        self.scale = 5

    def setscale(self):
        self.imageitem.resetTransform()
        self.imageitem.scale(self.scale,self.scale)

    def showlowres(self):
        #self.imgview.setImage(np.repeat(np.repeat(np.array(self.simg.thumbs.values()), 10, axis=0), 10, axis=1),
                              #xvals=self.simg.xvals)
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
        pipeline.variationoperators.experiment = self.parentwindow.experiment
        variation=self.simg.scan(self.operationindex)
        self.plotvariation(variation)

        for roi in self.viewbox.addedItems:
            #try:
                if hasattr(roi, 'isdeleting'):
                    if not roi.isdeleting:
                        roi=roi.getArrayRegion(np.ones_like(self.imgview.imageItem.image),self.imageitem)
                        variation=self.simg.scan(self.operationindex,roi)
                        self.plotvariation(variation,[0,255,255])

                    else:
                        self.viewbox.removeItem(roi)
            #except Exception as ex:
            #    print 'Warning: error displaying ROI variation.'
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

    def plotvariation(self,variation,color=None):
        if len(variation) == 0:
            return None

        if color is None:
            color = [255,255,255]

        # TODO: plot variation with indices, and skipped frames; skip None's


        variation = np.array(variation.items())
        # print variation
        variation = variation[variation[:, 0].argsort()]

        self.timeline.enableAutoScale()
        #self.timeruler = TimeRuler(pen=pg.mkPen('#FFA500', width=3), movable=True)

        print 'plottype:', type(variation[0, 1])
        if type(variation[0, 1]) is float or int:
            self.timeline.plot(variation[:, 0], variation[:, 1], pen=pg.mkPen(color=color))
        elif type(variation[0, 1]) is np.ndarray:
            self.timeline.addItem(pg.ImageItem(np.array(variation[:, 1])))


    def redrawframe(self, index, time, forcelow=False):
        key = round(time)
        self.dimg = self.simg.getDiffImage(key)
        self.redrawimage(forcelow=forcelow)

    def gotomax(self):
        pass
        #self.parentwindow.timeruler.setValue(np.argmax(self.variationy))
