from PySide import QtGui
from PySide import QtCore

class tomotoolbar(QtGui.QToolBar):

    def __init__(self):
        super(tomotoolbar, self).__init__()

        self.actionRun_SlicePreview = QtGui.QAction(self)
        icon17 = QtGui.QIcon()
        icon17.addPixmap(QtGui.QPixmap("gui/icons_50.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.actionRun_SlicePreview.setIcon(icon17)
        self.actionRun_SlicePreview.setToolTip('Slice preview')

        self.actionRun_3DPreview = QtGui.QAction(self)
        icon17 = QtGui.QIcon()
        icon17.addPixmap(QtGui.QPixmap("gui/icons_42.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.actionRun_3DPreview.setIcon(icon17)
        self.actionRun_3DPreview.setToolTip('3D preview')

        self.actionRun_FullRecon = QtGui.QAction(self)
        icon17 = QtGui.QIcon()
        icon17.addPixmap(QtGui.QPixmap("gui/icons_34.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.actionRun_FullRecon.setIcon(icon17)
        self.actionRun_FullRecon.setToolTip('Full reconstruction')

        self.actionShow_Mask = QtGui.QAction(self)
        self.actionShow_Mask.setCheckable(True)
        icon17 = QtGui.QIcon()
        icon17.addPixmap(QtGui.QPixmap("gui/icons_19.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.actionShow_Mask.setIcon(icon17)
        self.actionShow_Mask.setText('Show Mask')

        self.actionPolyMask = QtGui.QAction(self)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("gui/icons_05.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionPolyMask.setIcon(icon2)
        self.actionPolyMask.setText("Polygon Mask")

        self.actionMaskLoad = QtGui.QAction(self)
        icon7 = QtGui.QIcon()
        icon7.addPixmap(QtGui.QPixmap("gui/icons_08.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.actionMaskLoad.setIcon(icon7)
        self.actionMaskLoad.setText("Load Mask")

        self.actionMasking = QtGui.QAction(self)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap("gui/icons_03.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionMasking.setIcon(icon3)
        self.actionMasking.setToolTip('Masking')

        self.actionMasking = QtGui.QAction(self)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap("gui/icons_03.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionMasking.setIcon(icon3)

        maskmenu = QtGui.QMenu(self)
        maskmenu.addAction(self.actionShow_Mask)
        maskmenu.addAction(self.actionPolyMask)
        maskmenu.addAction(self.actionMaskLoad)
        toolbuttonMasking = QtGui.QToolButton(self)
        toolbuttonMasking.setDefaultAction(self.actionMasking)
        toolbuttonMasking.setMenu(maskmenu)
        toolbuttonMasking.setPopupMode(QtGui.QToolButton.InstantPopup)
        toolbuttonMaskingAction = QtGui.QWidgetAction(self)
        toolbuttonMaskingAction.setDefaultWidget(toolbuttonMasking)


        self.actionCenter = QtGui.QWidgetAction(self)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("gui/icons_28.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.actionCenter.setIcon(icon1)
        self.actionCenter.setToolTip('Overlay center of rotation detection')
        self.toolbuttonCenter = QtGui.QToolButton(parent=self)
        self.toolbuttonCenter.setPopupMode(QtGui.QToolButton.InstantPopup)
        self.actionCenter.setDefaultWidget(self.toolbuttonCenter)
        self.actionCenter.setCheckable(True)
        self.toolbuttonCenter.setDefaultAction(self.actionCenter)


        self.setIconSize(QtCore.QSize(32, 32))

        self.addAction(self.actionCenter)
        self.addAction(self.actionRun_FullRecon)
        self.addAction(self.actionRun_SlicePreview)
        self.addAction(self.actionRun_3DPreview)
        # self.addAction(toolbuttonMaskingAction)


    def connecttriggers(self, slicepreview, preview3D, fullrecon, center):
        self.actionRun_SlicePreview.triggered.connect(slicepreview)
        self.actionRun_3DPreview.triggered.connect(preview3D)
        self.actionRun_FullRecon.triggered.connect(fullrecon)
        self.actionCenter.toggled.connect(center)

