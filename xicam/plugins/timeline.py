import platform
from pipeline import msg

# Use NSURL as a workaround to pyside/Qt4 behaviour for dragging and dropping on OSx
op_sys = platform.system()
if op_sys == 'Darwin':
    try:
        from Foundation import NSURL
    except ImportError:
        msg.logMessage('NSURL not found. Drag and drop may not work correctly',msg.WARNING)


import base, viewer
from PySide import QtGui
import os
# from moviepy.editor import VideoClip
import numpy as np

import widgets
from pipeline import calibration
from xicam.widgets.NDTimelinePlotWidget import TimelinePlot


class TimelinePlugin(base.plugin):  ##### Inherit viewer instead!!!
    name = 'Timeline'
    sigUpdateExperiment = viewer.ViewerPlugin.sigUpdateExperiment

    def __init__(self, *args, **kwargs):
        self.centerwidget = QtGui.QTabWidget()
        self.centerwidget.currentChanged.connect(self.currentChanged)
        self.centerwidget.setDocumentMode(True)
        self.centerwidget.setTabsClosable(True)
        self.centerwidget.tabCloseRequested.connect(self.tabCloseRequested)

        self.bottomwidget = TimelinePlot()


        # Share right modes with viewer
        self.rightmodes = viewer.rightmodes

        self.toolbar = widgets.toolbar.difftoolbar()
        self.toolbar.connecttriggers(self.calibrate, self.centerfind, self.refinecenter, self.redrawcurrent,
                                     self.redrawcurrent, self.remeshmode, self.linecut, self.vertcut,
                                     self.horzcut, self.redrawcurrent, self.redrawcurrent, self.redrawcurrent,
                                     self.roi, self.arccut, self.polymask, process=self.process)
        super(TimelinePlugin, self).__init__(*args, **kwargs)

        # self.booltoolbar.actionTimeline.triggered.connect(self.openSelected)

        # DRAG-DROP
        self.centerwidget.setAcceptDrops(True)
        self.centerwidget.dragEnterEvent = self.dragEnterEvent
        self.centerwidget.dropEvent = self.dropEvent

    def dragEnterEvent(self, e):
        e.accept()
        # TODO: We should do something a bit less aggressive here!

    def dropEvent(self, e):
        if op_sys == 'Darwin':
            fnames = [str(NSURL.URLWithString_(str(url.toString())).filePathURL().path()) for url in
                      e.mimeData().urls()]
        else:
            fnames = e.mimeData().urls()
        e.accept()
        self.openfiles(fnames)



    def tabCloseRequested(self, index):
        self.centerwidget.widget(index).deleteLater()

    def getCurrentTab(self):
        return self.centerwidget.currentWidget().widget

    def currentImage(self):
        return self.getCurrentTab()

    def calibrate(self, algorithm=calibration.fourierAutocorrelation, calibrant='AgBh'):
        self.getCurrentTab().calibrate(algorithm, calibrant)

    def centerfind(self):
        self.getCurrentTab().centerfind()

    def refinecenter(self):
        self.getCurrentTab().refinecenter()

    def redrawcurrent(self):
        self.getCurrentTab().redrawimage()

    def remeshmode(self):
        self.getCurrentTab().redrawimage()
        self.getCurrentTab().replot()

    def linecut(self):
        self.getCurrentTab().linecut()

    def vertcut(self):
        self.getCurrentTab().verticalcut()

    def horzcut(self):
        self.getCurrentTab().horizontalcut()

    def roi(self):
        self.getCurrentTab().roi()

    def arccut(self):
        self.getCurrentTab().arccut()

    def polymask(self):
        self.getCurrentTab().polymask()

    def process(self):
        self.getCurrentTab().processtimeline()

    def makeVideo(self):
        pass  # disabled until solution is found for distributable version

        # fps, ok = QtGui.QInputDialog.getDouble(self.centerwidget, u'Enter frames per second:', u'Enter fps', value=24)
        #
        # def make_frame(t):
        # """ returns an image of the frame at time t """
        # # ... create the frame with any library
        #     img = self.getCurrentTab().simg[int(t * fps)].data
        #     img = np.rot90((np.log(img * (img > 0) + (img < 1))), 1)
        #     img = convertto8bit(img)
        #     return np.asarray(np.dstack((img, img, img)), dtype=np.uint8)
        #
        # if fps and ok:
        #     animation = VideoClip(make_frame, duration=len(self.getCurrentTab().simg) / fps)
        #
        #     # For the export, many options/formats/optimizations are supported
        #
        #     path, ok = QtGui.QFileDialog.getSaveFileName(self.centerwidget, 'Save Video', os.path.splitext(self.getCurrentTab().simg[0].filepath)[0] + '.mp4','*.mp4')
        #
        #     if path and ok:
        #         if os.path.splitext(path)[-1] == '.mp4':
        #             animation.write_videofile(path, fps=fps)  # export as video
        #         elif os.path.splitext(path)[-1] == '.gif':
        #             animation.write_gif(path, fps=fps)  # export as GIF (slow)
        #         else:
        #             animation.write_videofile(path, fps=fps)  # export as video
        #             print 'Error: Unrecognized extension...'


    def currentChanged(self, index):
        for tab in [self.centerwidget.widget(i) for i in range(self.centerwidget.count())]:
            tab.unload()
        self.centerwidget.currentWidget().load()

    def openfiles(self, files, operation=None, operationname=None):
        self.activate()
        widget = widgets.OOMTabItem(itemclass=widgets.timelineViewer, files=files, toolbar=self.toolbar)
        self.centerwidget.addTab(widget, 'Timeline: ' + os.path.basename(files[0]) + ', ...')
        self.centerwidget.setCurrentWidget(widget)
        self.getCurrentTab().sigAddTimelineData.connect(self.bottomwidget.addData)
        self.getCurrentTab().sigClearTimeline.connect(self.bottomwidget.clearData)


def convertto8bit(image):
    image *= (np.iinfo(np.uint8).max - 1) / float(np.max(image))
    return image.astype(np.uint8).copy()
