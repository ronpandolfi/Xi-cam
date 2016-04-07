from PySide import QtCore, QtGui
from PySide.QtUiTools import QUiLoader
from xicam.plugins import explorer
from pyqtgraph.parametertree import ParameterTree
from xicam import models
import toolbar as ttoolbar
import functiondata
import functionmanager

particlemenu = None
blankForm = None
leftwidget = None
centerwidget = None
rightwidget = None
bottomwidget = None
toolbar = None
propertytable = None
paramformstack = None
functionslist = None

class funcAction(QtGui.QAction):
    def __init__(self, func, subfunc, *args,**kwargs):
        super(funcAction, self).__init__(*args,**kwargs)
        self.func=func
        self.subfunc=subfunc
        self.triggered.connect(self.addFunction)
    def addFunction(self):
        functionmanager.addFunction(self.func,self.subfunc)

def load():
    global leftwidget, centerwidget, rightwidget, bottomwidget, blankForm, toolbar, propertytable, paramformstack, functionslist
    # Load the gui from file
    toolbar = ttoolbar.tomotoolbar()

    centerwidget = QtGui.QTabWidget()

    centerwidget.setDocumentMode(True)
    centerwidget.setTabsClosable(True)

    bottomwidget = None

    functionwidget = QUiLoader().load('gui/tomographyleft.ui')
    functionslist=functionwidget.functionsList

    addfunctionmenu = QtGui.QMenu()
    for func,subfuncs in functiondata.funcs.iteritems():
        funcmenu = QtGui.QMenu(func)
        addfunctionmenu.addMenu(funcmenu)
        for subfunc in subfuncs:
            funcaction=funcAction(func,subfunc,subfunc,funcmenu)
            funcmenu.addAction(funcaction)



    functionwidget.addFunctionButton.setMenu(addfunctionmenu)
    functionwidget.addFunctionButton.setPopupMode(QtGui.QToolButton.ToolButtonPopupMode.InstantPopup)

    functionwidget.clearButton.clicked.connect(functionmanager.clearFeatures)


    leftwidget =  QtGui.QSplitter(QtCore.Qt.Vertical)
    leftwidget.addWidget(functionwidget)
    fileexplorer = explorer.MultipleFileExplorer()
    leftwidget.addWidget(fileexplorer)




    rightwidget = QtGui.QWidget()
    l = QtGui.QVBoxLayout()
    l.setContentsMargins(0, 0, 0, 0)

    paramtree = ParameterTree()
    #configtree.setParameters(config.activeExperiment, showTop=False)
    #config.activeExperiment.sigTreeStateChanged.connect(self.sigUpdateExperiment)

    paramformstack = QtGui.QStackedWidget()
    paramformstack.addWidget(paramtree)
    l.addWidget(paramformstack)

    propertytable = QtGui.QTableView()

    propertytable.verticalHeader().hide()
    propertytable.horizontalHeader().hide()

    propertytable.horizontalHeader().setStretchLastSection(True)
    l.addWidget(propertytable)
    rightwidget.setLayout(l)

    blankForm = QtGui.QLabel('Select a function on the\nleft panel to edit...')
    blankForm.setSizePolicy(QtGui.QSizePolicy.Ignored, QtGui.QSizePolicy.Ignored)
    blankForm.setAlignment(QtCore.Qt.AlignCenter)
    showForm(blankForm)

    return leftwidget, centerwidget, rightwidget, bottomwidget, toolbar


def showForm(widget):
    paramformstack.addWidget(widget)
    paramformstack.setCurrentWidget(widget)