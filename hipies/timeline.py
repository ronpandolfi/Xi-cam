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

        self.getthumbnails()


        super(timelinetab, self).__init__(dimg, parentwindow)

        img = np.array(self.thumbnails)
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
        operationcombo.addItems(
            ['Chi Squared', 'Abs. difference', 'Norm. Abs. difference', 'Sum intensity', 'Norm. Abs. Diff. derivative'])
        #operationcombo.currentIndexChanged.connect(self.changetimelineoperation)
        opwidgetaction = QtGui.QWidgetAction(menu)
        opwidgetaction.setDefaultWidget(operationcombo)
        #need to connect it
        menu.addAction(opwidgetaction)


        # self.imgview.getHistogramWidget().item.setImageItem(self.highresimgitem)
        #self.imgview.getHistogramWidget().item.sigLevelChangeFinished.connect(self.updatelowresLUT)

    @debugtools.timeit
    def getthumbnails(self,):
        self.thumbnails = [dimg.thumbnail for dimg in self.simg]

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
        self.scale=10

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
        self.operationindex = index

    def cleartimeline(self):
        for item in self.timeline.items:
            if type(item) is pg.PlotDataItem:
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

        self.timeline.plot(variation[:, 0], variation[:, 1],pen=pg.mkPen(color=color))
        # self.parentwindow.timearrow = pg.ArrowItem(angle=-90, tipAngle=30, baseAngle=20,headLen=15,tailLen=None,brush=None,pen=pg.mkPen('#FFA500',width=3))
        #self.parentwindow.timeline.addItem(self.parentwindow.timearrow)
        #self.parentwindow.timearrow.setPos(0,self.variation[0])

    # def timerulermoved(self):
    #     pos = int(round(self.parentwindow.timeruler.value()))
    #
    #
    #     # snap to int
    #     self.parentwindow.timeruler.blockSignals(True)
    #     self.parentwindow.timeruler.setValue(pos)
    #     self.parentwindow.timeruler.blockSignals(False)
    #     #self.parentwindow.timearrow.setPos(pos,self.variation[pos])
    #
    #     if pos != self.previousPos:
    #         # print pos
    #         self.redrawframe()
    #     self.previousPos = pos

    # def timerulermouserelease(self, event):
    #     if event.button == QtCore.Qt.LeftButton:
    #         self.redrawimageFULL()

    def redrawframe(self, index, time, forcelow=False):
        key = round(time)
        self.dimg = self.simg.getDiffImage(key)
        self.redrawimage(forcelow=forcelow)

    def gotomax(self):
        pass
        #self.parentwindow.timeruler.setValue(np.argmax(self.variationy))

        # def roi(self):
        # if self.activeaction is None:  # If there is no active action
        #         self.activeaction = 'roi'
        #
        #         # Start with a box around the center
        #         left = self.dimg.experiment.getvalue('Center X') - 100
        #         right = self.dimg.experiment.getvalue('Center X') + 100
        #         up = self.dimg.experiment.getvalue('Center Y') - 100
        #         down = self.dimg.experiment.getvalue('Center Y') + 100
        #
        #         # Add ROI item to the image
        #         self.ROI = pg.PolyLineROI([[left, up], [left, down], [right, down], [right, up]], pen=(6, 9),
        #                                   closed=True)
        #         self.viewbox.addItem(self.ROI)
        #
        #         # Override the ROI's function to check if any points will be moved outside the boundary; False prevents move
        #         def checkPointMove(handle, pos, modifiers):
        #             p = self.viewbox.mapToView(pos)
        #             if 0 < p.y() < self.dimg.data.shape[0] and 0 < p.x() < self.dimg.data.shape[1]:
        #                 return True
        #             else:
        #                 return False
        #
        #         self.ROI.checkPointMove = checkPointMove
        #
        #     elif self.activeaction == 'roi':  # If the mask is completed
        #         self.activeaction = None
        #
        #         # Get the region of the image that was selected; unforunately the region is trimmed
        #         roiarea = self.ROI.getArrayRegion(np.ones_like(self.dimg.data.T), self.imageitem,
        #                                           returnMappedCoords=True)  # levels=(0, arr.max()
        #         # print maskedarea.shape
        #
        #         # Decide how much to left and top pad based on the ROI bounding rectangle
        #         boundrect = self.viewbox.itemBoundingRect(self.ROI)
        #         leftpad = boundrect.x()
        #         toppad = boundrect.y()
        #
        #         # Pad the mask so it has the same shape as the image
        #         roiarea = np.pad(roiarea, ((int(leftpad), 0), (int(toppad), 0)), mode='constant')
        #         roiarea = np.pad(roiarea, (
        #             (0, self.dimg.data.shape[0] - roiarea.shape[0]), (0, self.dimg.data.shape[1] - roiarea.shape[1])),
        #                          mode='constant')
        #
        #         # Add the masked area to the active mask
        #         self.simg.roi = roiarea
        #
        #         # Draw the overlay
        #         # self.maskoverlay()
        #
        #         # Remove the ROI
        #         self.viewbox.removeItem(self.ROI)
        #
        #         # Redo the integration
        #         self.rescan()


# class TimeRuler(pg.InfiniteLine):
#     sigRedrawFrame = QtCore.Signal(bool)
#
#     def __init__(self, pen, movable=True):
#         self.previousPos = None
#         super(TimeRuler, self).__init__(pen=pen, movable=movable)
#         self.previousPos = int(round(self.value()))
#         self.sigPositionChangeFinished.connect(self.endDrag)
#
#
#     def setPos(self, pos):
#         if type(pos) is pg.Point:
#             pos = pos.x()
#
#         pos = int(round(pos))
#
#         if pos != self.previousPos:
#             # snap to int
#             self.blockSignals(True)
#             super(TimeRuler, self).setPos(pos)
#             self.blockSignals(False)
#
#             self.sigRedrawFrame.emit(True)
#             self.previousPos = pos
#
#     def endDrag(self):
#         self.sigRedrawFrame.emit(False)
#
# class TimelineView
