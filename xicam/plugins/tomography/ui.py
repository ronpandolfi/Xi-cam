from PySide import QtCore, QtGui
from PySide.QtUiTools import QUiLoader
from functools import partial
from xicam import xglobals
from xicam.plugins import explorer, login
import pyqtgraph as pg
from pyqtgraph.parametertree import ParameterTree
from xicam import models
import toolbar as ttoolbar
import fdata
import fmanager

blankform = None
leftwidget = None
centerwidget = None
rightwidget = None
bottomwidget = None
toolbar = None
propertytable = None
paramformstack = None
functionslist = None
cor_spinBox = None


class funcAction(QtGui.QAction):
    def __init__(self, func, subfunc, *args,**kwargs):
        super(funcAction, self).__init__(*args,**kwargs)
        self.func=func
        self.subfunc=subfunc
        self.triggered.connect(self.addFunction)
    def addFunction(self):
        fmanager.add_function(self.func, self.subfunc)


def load():
    global leftwidget, centerwidget, rightwidget, bottomwidget, blankform, toolbar, propertytable, paramformstack, functionslist, cor_spinBox
    # Load the gui from file
    toolbar = ttoolbar.tomotoolbar()

    centerwidget = QtGui.QTabWidget()

    centerwidget.setDocumentMode(True)
    centerwidget.setTabsClosable(True)

    bottomwidget = None

    functionwidget = QUiLoader().load('gui/tomographyleft.ui')
    functionslist = functionwidget.functionsList

    addfunctionmenu = QtGui.QMenu()
    for func,subfuncs in fdata.funcs.iteritems():
        if len(subfuncs)>1 or func != subfuncs[0]:
            funcmenu = QtGui.QMenu(func)
            addfunctionmenu.addMenu(funcmenu)
            for subfunc in subfuncs:
                if isinstance(subfuncs, dict) and len(subfuncs[subfunc]) > 0:
                    optsmenu = QtGui.QMenu(subfunc)
                    funcmenu.addMenu(optsmenu)
                    for opt in subfuncs[subfunc]:
                        funcaction = funcAction(func, opt, opt, funcmenu)
                        optsmenu.addAction(funcaction)
                else:
                    funcaction=funcAction(func,subfunc,subfunc,funcmenu)
                    funcmenu.addAction(funcaction)
        elif len(subfuncs)==1:
            funcaction=funcAction(func,func,func,addfunctionmenu)
            addfunctionmenu.addAction(funcaction)

    functionwidget.addFunctionButton.setMenu(addfunctionmenu)
    functionwidget.addFunctionButton.setPopupMode(QtGui.QToolButton.ToolButtonPopupMode.InstantPopup)
    functionwidget.addFunctionButton.setArrowType(QtCore.Qt.NoArrow)

    #TODO find a way to share the base plugin loginwidget and fileexplorer
    leftwidget =  QtGui.QSplitter(QtCore.Qt.Vertical)
    leftwidget.addWidget(functionwidget)
    l = QtGui.QVBoxLayout()
    l.setContentsMargins(0, 0, 0, 0)
    loginwidget= login.LoginDialog()
    l.addWidget(loginwidget)
    fileexplorer =  explorer.MultipleFileExplorer()
    l.addWidget(fileexplorer)
    panelwidget = QtGui.QWidget()
    panelwidget.setLayout(l)
    leftwidget.addWidget(panelwidget)

    rightwidget = QtGui.QWidget()
    l = QtGui.QVBoxLayout()
    l.setContentsMargins(0, 0, 0, 0)

    paramtree = ParameterTree()

    paramformstack = QtGui.QStackedWidget()
    paramformstack.addWidget(paramtree)
    paramformstack.setMinimumHeight(200)
    l.addWidget(paramformstack)

    l1 = QtGui.QHBoxLayout()
    l1.setContentsMargins(0, 0, 0, 0)
    l1.addWidget(QtGui.QLabel('Center of rotation: '))
    cor_spinBox = QtGui.QDoubleSpinBox()
    cor_spinBox.setMaximum(9999)
    cor_spinBox.clear()
    l1.addWidget(cor_spinBox)

    l.insertLayout(2, l1)

    propertytable = pg.TableWidget() #QtGui.QTableView()
    propertytable.verticalHeader().hide()
    propertytable.horizontalHeader().setStretchLastSection(True)
    l.addWidget(propertytable)
    rightwidget.setLayout(l)

    blankform = QtGui.QLabel('Select a function on the\nleft panel to edit...')
    blankform.setSizePolicy(QtGui.QSizePolicy.Ignored, QtGui.QSizePolicy.Ignored)
    blankform.setAlignment(QtCore.Qt.AlignCenter)
    showform(blankform)

    loginwidget.loginClicked.connect(partial(xglobals.login, xglobals.spot_client))
    loginwidget.logoutClicked.connect(loginwidget.hide)
    loginwidget.logoutClicked.connect(fileexplorer.removeTabs)
    loginwidget.logoutClicked.connect(fileexplorer.enableActions)
    loginwidget.logoutClicked.connect(lambda: xglobals.logout(xglobals.spot_client, loginwidget.logoutSuccessful))
    loginwidget.sigLoggedIn.connect(xglobals.client_callback)

    fileexplorer.sigLoginRequest.connect(loginwidget.show)
    fileexplorer.sigLoginSuccess.connect(loginwidget.ui.user_box.setFocus)
    fileexplorer.sigLoginSuccess.connect(loginwidget.loginSuccessful)


    return leftwidget, centerwidget, rightwidget, bottomwidget, toolbar, functionwidget




def showform(widget):
    paramformstack.addWidget(widget)
    paramformstack.setCurrentWidget(widget)
