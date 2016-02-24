from collections import OrderedDict
from PySide import QtGui
import sys
from hipies import globals

modules = []
plugins = OrderedDict()



def initplugins(placeholders):
    import MOTD, viewer, timeline, library, fluctuationscattering, ipythonconsole, spoth5file

    global plugins, modules
    modules = [MOTD, viewer, timeline, library, ipythonconsole, fluctuationscattering, spoth5file]

    for module in modules:
        link = pluginlink(module, placeholders)
        link.enable()
        plugins[link.instance.name] = link
    globals.plugins = plugins


def buildactivatemenu(modewidget):
    menu = QtGui.QMenu('Plugins')
    for pluginlink in plugins.values():
        if pluginlink.instance.hidden:
            continue
        action = QtGui.QAction(pluginlink.name, menu)
        action.setCheckable(True)
        action.setChecked(True)
        action.toggled.connect(pluginlink.setEnabled)
        action.toggled.connect(modewidget.reload)
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
        del self.instance
        self.instance = None


    def enable(self):
        self.module = reload(sys.modules[self.modulename])
        self.plugin = self.module.plugin
        self.instance = self.plugin(self.placeholders)

    def setEnabled(self, enabled):
        if enabled:
            self.enable()
        else:
            self.disable()


    @property
    def enabled(self):
        return self.instance is not None

    @enabled.setter
    def enabled(self, enabled):
        if enabled:
            self.enable()
        else:
            self.disable()

    def activate(self):
        self.instance.activate()