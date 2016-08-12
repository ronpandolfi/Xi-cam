from PySide import QtGui
from PySide import QtCore

class Toolbar(QtGui.QToolBar):
    """
    QToolbar subclass used in Tomography plugin
    """

    def __init__(self):
        super(Toolbar, self).__init__()

        self.actionRun_SlicePreview = QtGui.QAction(self)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("gui/icons_50.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.actionRun_SlicePreview.setIcon(icon)
        self.actionRun_SlicePreview.setToolTip('Slice preview')

        self.actionRun_3DPreview = QtGui.QAction(self)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("gui/icons_42.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.actionRun_3DPreview.setIcon(icon)
        self.actionRun_3DPreview.setToolTip('3D preview')

        self.actionRun_FullRecon = QtGui.QAction(self)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("gui/icons_34.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.actionRun_FullRecon.setIcon(icon)
        self.actionRun_FullRecon.setToolTip('Full reconstruction')

        self.actionPolyMask = QtGui.QAction(self)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("gui/icons_05.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionPolyMask.setIcon(icon)
        self.actionPolyMask.setText("Polygon mask")

        self.actionCircMask = QtGui.QAction(self)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("gui/icons_05.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.actionCircMask.setIcon(icon)
        self.actionCircMask.setText("Circular mask")

        self.actionRectMask = QtGui.QAction(self)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("gui/icons_05.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionRectMask.setIcon(icon)
        self.actionRectMask.setText('Rectangular mask')

        self.actionMask = QtGui.QAction(self)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("gui/icons_03.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.actionMask.setIcon(icon)

        maskmenu = QtGui.QMenu(self)
        maskmenu.addAction(self.actionRectMask)
        maskmenu.addAction(self.actionCircMask)
        maskmenu.addAction(self.actionPolyMask)
        toolbuttonMasking = QtGui.QToolButton(self)
        toolbuttonMasking.setDefaultAction(self.actionMask)
        toolbuttonMasking.setMenu(maskmenu)
        toolbuttonMasking.setPopupMode(QtGui.QToolButton.InstantPopup)
        toolbuttonMaskingAction = QtGui.QWidgetAction(self)
        toolbuttonMaskingAction.setDefaultWidget(toolbuttonMasking)


        self.actionCenter = QtGui.QWidgetAction(self)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("gui/icons_28.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.actionCenter.setIcon(icon)
        self.actionCenter.setToolTip('Overlay center of rotation detection')
        self.toolbuttonCenter = QtGui.QToolButton(parent=self)
        self.toolbuttonCenter.setPopupMode(QtGui.QToolButton.InstantPopup)
        self.actionCenter.setDefaultWidget(self.toolbuttonCenter)
        self.actionCenter.setCheckable(True)
        self.toolbuttonCenter.setDefaultAction(self.actionCenter)

        # TODO working on ROI Selection TOOL
        self.actionROI = QtGui.QAction(self)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("gui/icons_60.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionROI.setIcon(icon)
        self.actionROI.setToolTip('Selecte region of interest')

        self.setIconSize(QtCore.QSize(32, 32))

        self.addAction(self.actionRun_FullRecon)
        self.addAction(self.actionRun_SlicePreview)
        self.addAction(self.actionRun_3DPreview)
        self.addAction(self.actionCenter)
        # self.addAction(self.actionROI)
        # self.addAction(toolbuttonMaskingAction)


    def connecttriggers(self, slicepreview, preview3D, fullrecon, center):
        """
        Connect toolbar action signals to give slots

        Parameters
        ----------
        slicepreview : QtCore.Slot
            Slot to connect actionRun_SlicePreview
        preview3D : QtCore.Slot
            Slot to connect actionRun_3DPreview
        fullrecon : QtCore.Slot
            Slot to connect actionRun_FullRecon
        center : QtCore.Slot
            Slot to connect actionCenter

        """
        self.actionRun_SlicePreview.triggered.connect(slicepreview)
        self.actionRun_3DPreview.triggered.connect(preview3D)
        self.actionRun_FullRecon.triggered.connect(fullrecon)
        self.actionCenter.toggled.connect(center)

