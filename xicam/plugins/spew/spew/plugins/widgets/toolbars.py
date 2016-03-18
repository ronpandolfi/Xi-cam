# -*- coding: utf-8 -*-
"""
@author:
"""
from PySide import QtGui, QtCore

QtCore.Signal = QtCore.Signal
QtCore.Slot = QtCore.Slot


class TomoToolbar(QtGui.QToolBar):
    """
    Toolbar that exposes most of tomopy functionality
    """

    def __init__(self, parent):
        super(TomoToolbar, self).__init__()
        self.parent = parent

        # Create actions in toolbar
        self.actionROI = QtGui.QAction(self)
        icon0 = QtGui.QIcon()
        icon0.addPixmap(QtGui.QPixmap('gui/spew/roi.png'), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionROI.setIcon(icon0)
        self.actionROI.setObjectName('actionROI')
        self.actionROI.setToolTip('Select region of interest')
        self.actionRecon = QtGui.QAction(self)
        self.actionNorm = QtGui.QAction(self)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap('gui/spew/norm.png'), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionNorm.setIcon(icon1)
        self.actionNorm.setObjectName('actionNorm')
        self.actionNorm.setToolTip('Normalize raw data')
        self.actionCOR = QtGui.QAction(self)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap('gui/spew/cor.png'), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionCOR.setIcon(icon2)
        self.actionCOR.setObjectName('actionCOR')
        self.actionCOR.setToolTip('Find center of rotation')
        self.actionStripe = QtGui.QAction(self)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap('gui/spew/stripes.png'), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionStripe.setIcon(icon3)
        self.actionStripe.setObjectName('actionStripe')
        self.actionStripe.setToolTip('Remove stripe artifacts from sinograms')
        self.actionRing = QtGui.QAction(self)
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap('gui/spew/rings.png'), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionRing.setIcon(icon4)
        self.actionRing.setObjectName('actionRing')
        self.actionRing.setToolTip('Remove ring artifacts from reconstruction')
        self.actionPhase = QtGui.QAction(self)
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap('gui/spew/phase.png'), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionPhase.setIcon(icon5)
        self.actionPhase.setObjectName('actionPhase')
        self.actionPhase.setToolTip('Retrieve phase from phase-contrast data')
        self.actionFilter = QtGui.QAction(self)
        icon6 = QtGui.QIcon()
        icon6.addPixmap(QtGui.QPixmap('gui/spew/filter.png'), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionFilter.setIcon(icon6)
        self.actionFilter.setObjectName('actionFilter')
        self.actionFilter.setToolTip('Apply filter to data')
        icon7 = QtGui.QIcon()
        icon7.addPixmap(QtGui.QPixmap('gui/spew/recon.png'), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionRecon.setIcon(icon7)
        self.actionRecon.setObjectName('actionRecon')
        self.actionRecon.setToolTip('Reconstruct dataset')
        self.actionMask = QtGui.QAction(self)
        icon8 = QtGui.QIcon()
        icon8.addPixmap(QtGui.QPixmap('gui/spew/mask.png'), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionMask.setIcon(icon8)
        self.actionMask.setObjectName('actionMask')
        self.actionMask.setToolTip('Apply circular mask')
        self.actionPreview = QtGui.QAction(self)
        icon9 = QtGui.QIcon()
        icon9.addPixmap(QtGui.QPixmap('gui/spew/preview.png'), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionPreview.setIcon(icon9)
        self.actionPreview.setObjectName('actionPreview')
        self.actionPreview.setToolTip('Preview dataset')
        self.actionWizard = QtGui.QAction(self)
        icon10 = QtGui.QIcon()
        icon10.addPixmap(QtGui.QPixmap('gui/spew/wizard.png'), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionWizard.setIcon(icon10)
        self.actionWizard.setObjectName('actionWizard')
        self.actionWizard.setToolTip('Reconstruction wizard')

        self.addAction(self.actionPreview)
        self.addAction(self.actionROI)
        self.addAction(self.actionNorm)
        self.addAction(self.actionCOR)
        self.addAction(self.actionStripe)
        self.addAction(self.actionRing)
        self.addAction(self.actionPhase)
        self.addAction(self.actionFilter)
        self.addAction(self.actionRecon)
        self.addAction(self.actionMask)
        self.addAction(self.actionWizard)
        self.unimplemented_actions = (self.actionPreview, self.actionROI, self.actionNorm, self.actionCOR,
                                      self.actionStripe, self.actionRing, self.actionPhase, self.actionFilter,
                                      self.actionRecon, self.actionMask)

        self.setIconSize(QtCore.QSize(32, 32))

    def connect_triggers(self, wizard):# preview, roi, norm, cor, stripe, ring, phase, filter, recon, mask, wizard):
        self.actionWizard.triggered.connect(wizard)
        for action in self.unimplemented_actions:
            action.triggered.connect(self.unimplemented_message)

    def unimplemented_message(self):
        msg_box = QtGui.QMessageBox.warning(self.parent.parent, 'Not implemented',
                                            'This action has not been implemented yet.\nThe SPEW crew is working '
                                            'hard to have it working soon!')
        return


class ExplorerToolbar(QtGui.QToolBar):
    """
    Toolbar for file movement actions
    """

    def __init__(self, parent=None):
        super(ExplorerToolbar, self).__init__(parent)
        self.parent = parent

        # Create actions in toolbar
        self.actionOpen = QtGui.QAction(self)
        icon0 = QtGui.QIcon()
        icon0.addPixmap(QtGui.QPixmap('gui/spew/open.png'), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionOpen.setIcon(icon0)
        self.actionOpen.setObjectName('actionOpen')
        self.actionOpen.setToolTip('Open dataset')
        self.actionDelete = QtGui.QAction(self)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap('gui/spew/delete.png'), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionDelete.setIcon(icon1)
        self.actionDelete.setObjectName('actionDelete')
        self.actionDelete.setToolTip('Delete dataset')
        self.actionUpload = QtGui.QAction(self)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap('gui/spew/upload.png'), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionUpload.setIcon(icon2)
        self.actionUpload.setObjectName('actionUplaod')
        self.actionUpload.setToolTip('Upload dataset')
        self.actionTransfer = QtGui.QAction(self)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap('gui/spew/transfer.png'), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionTransfer.setIcon(icon3)
        self.actionTransfer.setObjectName('actionTransfer')
        self.actionTransfer.setToolTip('Transfer dataset')
        self.actionDownload = QtGui.QAction(self)
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap('gui/spew/download.png'), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionDownload.setIcon(icon4)
        self.actionDownload.setObjectName('actionDownload')
        self.actionDownload.setToolTip('Download dataset')

        self.addAction(self.actionOpen)
        self.addAction(self.actionDelete)
        self.addAction(self.actionUpload)
        self.addAction(self.actionDownload)
        self.addAction(self.actionTransfer)

        self.setIconSize(QtCore.QSize(32, 32))

    def connect_triggers(self, open, delete, upload, download, transfer):
        self.actionOpen.triggered.connect(open)
        self.actionDelete.triggered.connect(delete)
        self.actionUpload.triggered.connect(upload)
        self.actionDownload.triggered.connect(download)
        self.actionTransfer.triggered.connect(transfer)


    def unimplemented_message(self):
        msg_box = QtGui.QMessageBox.warning(self.parent, 'Not implemented',
                                            'This action has not been implemented yet.\nThe SPEW crew is working '
                                            'hard to have it working soon!')
        return
