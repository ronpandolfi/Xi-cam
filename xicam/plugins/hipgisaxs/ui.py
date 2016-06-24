from PySide import QtCore, QtGui
from PySide.QtUiTools import QUiLoader

particlemenu = None
blankForm = None
leftwidget = None
centerwidget = None
rightwidget = None


def load():
    global leftwidget, centerwidget, rightwidget, blankForm
    # Load the gui from file
    guiloader = QUiLoader()
    f = QtCore.QFile("gui/hipgisaxsleftwidget.ui")
    f.open(QtCore.QFile.ReadOnly)
    leftwidget = guiloader.load(f)
    f.close()

    rightwidget = QtGui.QStackedWidget()
    centerwidget = QtGui.QSplitter(QtCore.Qt.Vertical)

    blankForm = QtGui.QLabel('Select a feature on the left panel to edit...')
    blankForm.setSizePolicy(QtGui.QSizePolicy.Ignored, QtGui.QSizePolicy.Ignored)
    blankForm.setAlignment(QtCore.Qt.AlignCenter)
    showForm(blankForm)

    return leftwidget, centerwidget, rightwidget


def showForm(form):
    print 'form:', form
    rightwidget.addWidget(form)
    rightwidget.setCurrentWidget(form)