from PySide import QtGui
from PySide import QtCore

class tomotoolbar(QtGui.QToolBar):

    def __init__(self):
        super(tomotoolbar, self).__init__()

        self.actionShow_Mask = QtGui.QAction(self)
        self.actionShow_Mask.setCheckable(True)
        icon17 = QtGui.QIcon()
        icon17.addPixmap(QtGui.QPixmap("gui/icons_20.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        icon17.addPixmap(QtGui.QPixmap("gui/icons_19.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.actionShow_Mask.setIcon(icon17)
        self.actionShow_Mask.setObjectName("actionShow_Mask")
        self.actionShow_Mask.setText('Show Mask')

        self.actionPolyMask = QtGui.QAction(self)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("gui/icons_05.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionPolyMask.setIcon(icon2)
        self.actionPolyMask.setText("Polygon Mask")
        self.actionPolyMask.setObjectName("actionPolyMask")

        self.actionMaskLoad = QtGui.QAction(self)
        icon7 = QtGui.QIcon()
        icon7.addPixmap(QtGui.QPixmap("gui/icons_08.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.actionMaskLoad.setIcon(icon7)
        self.actionMaskLoad.setText("Load Mask")
        self.actionMaskLoad.setObjectName("actionMaskLoad")

        self.actionMasking = QtGui.QAction(self)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap("gui/icons_03.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionMasking.setIcon(icon3)
        self.actionMasking.setToolTip('Masking')
        self.actionMasking.setObjectName("actionMasking")

        self.actionMasking = QtGui.QAction(self)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap("gui/icons_03.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionMasking.setIcon(icon3)
        self.actionMasking.setObjectName("actionMasking")

        maskmenu = QtGui.QMenu()
        maskmenu.addAction(self.actionShow_Mask)
        maskmenu.addAction(self.actionPolyMask)
        maskmenu.addAction(self.actionMaskLoad)
        toolbuttonMasking = QtGui.QToolButton()
        toolbuttonMasking.setDefaultAction(self.actionMasking)
        toolbuttonMasking.setMenu(maskmenu)
        toolbuttonMasking.setPopupMode(QtGui.QToolButton.InstantPopup)
        toolbuttonMaskingAction = QtGui.QWidgetAction(self)
        toolbuttonMaskingAction.setDefaultWidget(toolbuttonMasking)

        centermenu = QtGui.QMenu()
        icon1 = QtGui.QIcon()
        icon2 = QtGui.QIcon()
        icon3 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("gui/icons_28.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        icon2.addPixmap(QtGui.QPixmap("gui/icons_28.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        icon3.addPixmap(QtGui.QPixmap("gui/icons_28.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.actionAutoCenter=QtGui.QAction(icon1,'Auto Center',centermenu)
        self.actionSemiAutoCenter=QtGui.QAction(icon2,'Semi-Auto Center',centermenu)
        self.actionManualCenter=QtGui.QAction(icon2,'Manual Center',centermenu)
        centermenu.addActions([self.actionAutoCenter, self.actionSemiAutoCenter, self.actionManualCenter])
        toolbuttonCenter = QtGui.QToolButton()
        toolbuttonCenter.setDefaultAction(self.actionAutoCenter)
        toolbuttonCenter.setMenu(centermenu)
        toolbuttonCenter.setPopupMode(QtGui.QToolButton.InstantPopup)
        toolbuttonCenterAction = QtGui.QWidgetAction(self)
        toolbuttonCenterAction.setDefaultWidget(toolbuttonCenter)




        # Normalize
        ## ROI Normalize
        ## Normalize
        ## Flat-field normalize
        ## Normalize background
        # Find center
        ## Manual
        ## Automatic
        ### Phase correlation
        ### Nelder-Mead
        ### Vo
        # Stripe Correction
        ## Fourier-wavelet
        ## Smoothing filter
        ## Titarenko
        # Ring Correction
        # Phase retrieval
        # Filters
        ## Gaussian
        ## Mean
        ## Median
        ## Sobel
        # Mask



        self.setIconSize(QtCore.QSize(32, 32))

        self.addAction(toolbuttonMaskingAction)
        self.addAction(toolbuttonCenterAction)



    def connecttriggers(self):
        pass
