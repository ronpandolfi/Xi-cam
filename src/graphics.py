import pyqtgraph as pg
import numpy as np

import integration
import center_approx
from PySide.QtGui import QVBoxLayout
from PySide.QtGui import QSplitter
from PySide.QtGui import QWidget
from PySide.QtCore import Qt
from PySide.QtGui import QAction


class imageTab(QWidget):
    def __init__(self, imgdata, experiment):
        '''
        A tab containing an imageview and plotview set in a splitter. Also manages functionality connected to a specific tab (masking/integration)
        :param imgdata:
        :param experiment:
        :return:
        '''
        super(imageTab, self).__init__()

        # Save image data and the experiment
        self.imgdata = imgdata
        self.experiment = experiment

        # Immediately mask any negative pixels
        self.experiment.mask = imgdata < 0

        # For storing what action is active (mask/circle fit...)
        self.activeaction = None

        # Make an imageview for the image
        self.imgview = pg.ImageView(self)
        self.imgview.setImage(imgdata)
        self.imgview.autoRange()
        self.imageitem = self.imgview.getImageItem()
        self.viewbox = self.imgview.getView()

        # Add a thin border to the image so it is visible on black background
        self.imageitem.border = pg.mkPen('w')

        # Make a splitter for the tab (inside layout); put the image view in it
        self.Splitter = QSplitter(Qt.Vertical)
        self.Splitter.addWidget(self.imgview)

        # Make a layout for the tab
        self.Layout = QVBoxLayout()
        self.Layout.addWidget(self.Splitter)
        self.setLayout(self.Layout)

        # Add the Log Intensity check button to the context menu and wire up
        self.imgview.buildMenu()
        menu = self.viewbox.menu
        self.actionLogIntensity = QAction('Log Intensity', menu, checkable=True)
        self.actionLogIntensity.triggered.connect(self.logintensity)
        menu.addAction(self.actionLogIntensity)
        self.imgview.buildMenu()

        # Add a placeholder image item for the mask to the viewbox
        self.maskimage = pg.ImageItem(opacity=.25)
        self.viewbox.addItem(self.maskimage)

        # Add a plot widget to the splitter for integration
        integrationwidget = pg.PlotWidget()
        self.integration = integrationwidget.getPlotItem()
        self.Splitter.addWidget(integrationwidget)

        ##
        self.findcenter()
        self.radialintegrate()
        ##

        # Mark the center
        self.centerplot = pg.ScatterPlotItem([self.experiment.getvalue('Center Y')],
                                             [self.experiment.getvalue('Center X')], pen=None, symbol='o')
        self.viewbox.addItem(self.centerplot)


    def logintensity(self):
        # When the log intensity button toggles, switch the log scaling on the image
        if self.actionLogIntensity.isChecked():
            self.imgview.setImage(np.log(self.imgdata * (self.imgdata > 0) + (self.imgdata < 1)))
        else:
            self.imgview.setImage(self.imgdata)

    def findcenter(self):
        # Auto find the beam center
        [x, y] = center_approx.center_approx(self.imgdata)

        # Set the center in the experiment
        self.experiment.setValue('Center X', x)
        self.experiment.setValue('Center Y', y)

    def radialintegrate(self):
        # Radial integraion
        q, radialprofile = integration.radialintegrate(self.imgdata, self.experiment, mask=self.experiment.mask)

        # Clear previous plot and replot
        self.integration.clear()
        self.integration.plot(q, radialprofile)


    def polymask(self):
        if self.activeaction is None:  # If there is no active action
            self.activeaction = 'polymask'

            # Start with a box around the center
            left = self.experiment.getvalue('Center Y') - 100
            right = self.experiment.getvalue('Center Y') + 100
            up = self.experiment.getvalue('Center X') - 100
            down = self.experiment.getvalue('Center X') + 100

            # Add ROI item to the image
            self.maskROI = pg.PolyLineROI([[left, up], [left, down], [right, down], [right, up]], pen=(6, 9),
                                          closed=True)
            self.viewbox.addItem(self.maskROI)

            # Override the ROI's function to check if any points will be moved outside the boundary; False prevents move
            def checkPointMove(handle, pos, modifiers):
                p = self.viewbox.mapToView(pos)
                if 0 < p.x() < self.imgdata.shape[0] and 0 < p.y() < self.imgdata.shape[1]:
                    return True
                else:
                    return False

            self.maskROI.checkPointMove = checkPointMove

        elif self.activeaction == 'polymask':  # If the mask is completed
            self.activeaction = None

            # Get the region of the image that was selected; unforunately the region is trimmed
            maskedarea = self.maskROI.getArrayRegion(np.ones_like(self.imgdata), self.imageitem,
                                                     returnMappedCoords=True)  # levels=(0, arr.max()

            # Decide how much to left and top pad based on the ROI bounding rectangle
            boundrect = self.viewbox.itemBoundingRect(self.maskROI)
            leftpad = boundrect.x()
            toppad = boundrect.y()

            # Pad the mask so it has the same shape as the image
            maskedarea = np.pad(maskedarea, ((int(leftpad), 0), (int(toppad), 0)), mode='constant')
            maskedarea = np.pad(maskedarea, (
                (0, self.imgdata.shape[0] - maskedarea.shape[0]), (0, self.imgdata.shape[1] - maskedarea.shape[1])),
                                mode='constant')

            # Add the masked area to the active mask
            self.addtomask(maskedarea)

            # Draw the overlay
            self.maskoverlay()

            # Remove the ROI
            self.viewbox.removeItem(self.maskROI)

            # Redo the integration
            self.radialintegrate()

    def addtomask(self, maskedarea):
        # If the mask is empty, set the mask to the new masked area
        if self.experiment.mask is None:
            self.experiment.mask = maskedarea.astype(np.int)
        else:  # Otherwise, bitwise or it with the current mask
            # print(self.experiment.mask,maskedarea)
            self.experiment.mask = np.bitwise_or(self.experiment.mask, maskedarea.astype(np.int))

    def maskoverlay(self):
        # Draw the mask as a red channel image with an alpha mask
        self.maskimage.setImage(np.dstack((
            self.experiment.mask, np.zeros_like(self.experiment.mask), np.zeros_like(self.experiment.mask),
            self.experiment.mask)), opacity=.25)

    def viewmask(self):
        view = pg.ImageView()
        view.setImage(self.experiment.mask)
        view.show()




