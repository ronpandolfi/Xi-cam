from collections import OrderedDict
from PySide import QtGui
import sys

modules = []
plugins = OrderedDict()


def initplugins(placeholders):
    import viewer, timeline, library, fluctuationscattering, ipythonconsole, spoth5file

    global plugins, modules
    modules = [viewer, timeline, library, ipythonconsole, fluctuationscattering, spoth5file]

    for module in modules:
        link = pluginlink(module, placeholders)
        link.enable()
        plugins[link.instance.name] = link


def buildactivatemenu(modewidget):
    menu = QtGui.QMenu('Plugins')
    for pluginlink in plugins.values():
        action = QtGui.QAction(pluginlink.name, menu)
        action.setCheckable(True)
        action.setChecked(True)
        action.toggled.connect(pluginlink.setEnabled)
        action.changed.connect(modewidget.reload)
        menu.addAction(action)
    return menu


class pluginlink():
    def __init__(self, module, placeholders):
        self.plugin = module.plugin
        self.modulename = module.__name__
        self.module = module
        self.instance = None
        self.placeholders = placeholders
        self.name = self.plugin.name

    def disable(self):
        self.instance = None


    def enable(self):
        self.module = reload(sys.modules[self.modulename])
        self.plugin = self.module.plugin
        self.instance = self.plugin(self.placeholders)

    def setEnabled(self, enabled):
        self.enabled = enabled


    @property
    def enabled(self):
        return self.instance is not None

    @enabled.setter
    def enabled(self, enabled):
        if enabled:
            self.enable()
        else:
            self.disable()