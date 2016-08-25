from PySide import QtGui, QtCore
from pyqtgraph.parametertree import ParameterTree
from functools import partial
from PySide.QtUiTools import QUiLoader
from psutil import cpu_count
import pyqtgraph as pg


class workflowEditorWidget(QtGui.QSplitter):
    def __init__(self):
        super(workflowEditorWidget, self).__init__()
        self.functionwidget = QUiLoader().load('/home/rp/PycharmProjects/xicam/gui/tomographyleft.ui')
        self.functionwidget.functionsList.setAlignment(QtCore.Qt.AlignBottom)

        self.functionwidget.addFunctionButton.setToolTip('Add function to pipeline')
        self.functionwidget.clearButton.setToolTip('Clear pipeline')
        self.functionwidget.fileButton.setToolTip('Save/Load pipeline')
        self.functionwidget.moveDownButton.setToolTip('Move selected function down')
        self.functionwidget.moveUpButton.setToolTip('Move selected function up')

        self.addfunctionmenu = QtGui.QMenu()
        self.functionwidget.addFunctionButton.setMenu(self.addfunctionmenu)
        self.functionwidget.addFunctionButton.setPopupMode(QtGui.QToolButton.ToolButtonPopupMode.InstantPopup)
        self.functionwidget.addFunctionButton.setArrowType(QtCore.Qt.NoArrow)

        filefuncmenu = QtGui.QMenu()
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("gui/icons_55.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.openaction = QtGui.QAction(icon, 'Open', filefuncmenu)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("gui/icons_59.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.saveaction = QtGui.QAction(icon, 'Save', filefuncmenu)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("gui/icons_56.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.refreshaction = QtGui.QAction(icon, 'Reset', filefuncmenu)
        filefuncmenu.addActions([self.openaction, self.saveaction, self.refreshaction])

        self.functionwidget.fileButton.setMenu(filefuncmenu)
        self.functionwidget.fileButton.setPopupMode(QtGui.QToolButton.ToolButtonPopupMode.InstantPopup)
        self.functionwidget.fileButton.setArrowType(QtCore.Qt.NoArrow)

        paramtree = ParameterTree()
        self.nodeEditor = QtGui.QStackedWidget()
        self.nodeEditor.addWidget(paramtree)
        self.addWidget(self.nodeEditor)
        self.addWidget(self.functionwidget)

    def connectTriggers(self, open, save, reset, moveup, movedown, clear):
        """
        Connect leftwidget (function mangement buttons) triggers to corresponding slots

        Parameters
        ----------
        open : QtCore.Slot
            Slot to handle signal from open button
        save QtCore.Slot
            Slot to handle signal from save button
        reset QtCore.Slot
            Slot to handle signal from reset button
        moveup QtCore.Slot
            Slot to handle signal to move a function widget upwards
        movedown QtCore.Slot
            Slot to handle signal to move a function widget downwards
        clear QtCore.Slot
            Slot to handle signal from clear button
        """

        self.openaction.triggered.connect(open)
        self.saveaction.triggered.connect(save)
        self.refreshaction.triggered.connect(reset)
        self.functionwidget.moveDownButton.clicked.connect(moveup)
        self.functionwidget.moveUpButton.clicked.connect(movedown)
        self.functionwidget.clearButton.clicked.connect(clear)

    def build_function_menu(menu, functree, functiondata, actionslot):
        """
        Builds the function menu's and submenu's anc connects them to the corresponding slot to add them to the workflow
        pipeline

        Parameters
        ----------
        menu : QtGui.QMenu
            Menu object to populate with submenu's and actions
        functree : dict
            Dictionary specifying the depth levels of functions. See functions.yml entry "Functions"
        functiondata : dict
            Dictionary with function information. See function_names.yml
        actionslot : QtCore.Slot
            slot where the function action triggered signal shoud be connected
        """

        for func, subfuncs in functree.iteritems():
            if len(subfuncs) > 1 or func != subfuncs[0]:
                funcmenu = QtGui.QMenu(func)
                menu.addMenu(funcmenu)
                for subfunc in subfuncs:
                    if isinstance(subfuncs, dict) and len(subfuncs[subfunc]) > 0:
                        optsmenu = QtGui.QMenu(subfunc)
                        funcmenu.addMenu(optsmenu)
                        for opt in subfuncs[subfunc]:
                            funcaction = QtGui.QAction(opt, funcmenu)
                            try:
                                funcaction.triggered.connect(partial(actionslot, func, opt,
                                                                     reconpkg.packages[functiondata[opt][1]]))
                                optsmenu.addAction(funcaction)
                            except KeyError:
                                pass
                    else:
                        funcaction = QtGui.QAction(subfunc, funcmenu)
                        try:
                            funcaction.triggered.connect(partial(actionslot, func, subfunc,
                                                                 reconpkg.packages[functiondata[subfunc][1]]))
                            funcmenu.addAction(funcaction)
                        except KeyError:
                            pass
            elif len(subfuncs) == 1:
                try:
                    funcaction = QtGui.QAction(func, menu)
                    funcaction.triggered.connect(
                        partial(actionslot, func, func, reconpkg.packages[functiondata[func][1]]))
                    menu.addAction(funcaction)
                except KeyError:
                    pass


if __name__ == '__main__':

    app = QtGui.QApplication([])

    w = workflowEditorWidget()
    w.show()

    import sys

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
