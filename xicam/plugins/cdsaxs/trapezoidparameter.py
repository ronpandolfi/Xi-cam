from pyqtgraph.parametertree import parameterTypes as pTypes
from pyqtgraph.parametertree import Parameter, ParameterItem
from PySide import QtGui,QtCore

_fromUtf8 = lambda s:s

class TrapezoidAnglesWidget(QtGui.QWidget):
    sigValueChanged = QtCore.Signal()
    sigChanged = sigValueChanged

    def __init__(self):
        super(TrapezoidAnglesWidget, self).__init__()

        self.verticalLayout = QtGui.QVBoxLayout(self)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)

        self.toolbar = QtGui.QToolBar()
        self.toolbar.addAction("+",self.addSpinBox)
        self.toolbar.addAction("-",self.deleteSpinBox)
        self.verticalLayout.addWidget(self.toolbar)

        self.spinboxes=[QtGui.QSpinBox()]
        self.verticalLayout.addWidget(self.spinboxes[0])


    def addSpinBox(self):
        self.spinboxes.append(QtGui.QSpinBox())
        self.verticalLayout.addWidget(self.spinboxes[-1])

    def deleteSpinBox(self):
        self.spinboxes.pop().setParent(None)


    def value(self):
        return tuple(spinbox.value() for spinbox in self.spinboxes)

    def setValue(self, v):
        if v is None: return
        for spinbox,value in zip(self.spinboxes,v):
            spinbox.setValue(value)

    def setEnabled(self, enabled):
        for spinbox in self.spinboxes:
            spinbox.setEnabled(enabled)

class TrapezoidAnglesWidgetParameterItem(pTypes.WidgetParameterItem):
    def __init__(self, param, depth):
        super(TrapezoidAnglesWidgetParameterItem, self).__init__(param, depth)
        self.subItem = QtGui.QTreeWidgetItem()
        self.addChild(self.subItem)

    def makeWidget(self, cls = TrapezoidAnglesWidget):
        w = cls()
        opts = self.param.opts
        value = opts.get('value', None)
        if value is not None:
            w.setValue(value)
        else:
            w.setValue([90])

        self.value = w.value
        self.setValue = w.setValue

        self.widget = w

        return w

    def treeWidgetChanged(self):
        ## TODO: fix so that superclass method can be called
        ## (WidgetParameter should just natively support this style)
        # WidgetParameterItem.treeWidgetChanged(self)
        self.treeWidget().setFirstItemColumnSpanned(self.subItem, True)
        self.treeWidget().setItemWidget(self.subItem, 0, self.widget)

        # for now, these are copied from ParameterItem.treeWidgetChanged
        self.setHidden(not self.param.opts.get('visible', True))
        self.setExpanded(self.param.opts.get('expanded', True))

class TrapezoidAnglesWidgetParameter(Parameter):
    itemClass = TrapezoidAnglesWidgetParameterItem

    def __init__(self, *args, **kwargs):
        super(TrapezoidAnglesWidgetParameter, self).__init__(*args, **kwargs)

    def defaultValue(self):
        return (90)