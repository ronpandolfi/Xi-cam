from collections import OrderedDict
from PySide import QtGui
import sys
from xicam import xglobals
import importlib
from pipeline import msg
import os
import pkgutil

modules = []
plugins = OrderedDict()

disabledatstart = ['FXS', 'SPOTH5', 'Library', 'XAS']


def initplugins(placeholders):
    global plugins, modules

    packages = pkgutil.iter_modules(__path__, '.')
    print 'packages:',packages
    packages = [pkg for pkg in packages if pkg[1] not in ['.widgets', '.login', '.base', '.explorer', '.__init__']]
    print 'packages:', packages

    for importer, modname, ispkg in packages:
        try:
            print "Found plugin %s (is a package: %s)" % (modname, ispkg)
            modules.append(importlib.import_module(modname,'xicam.plugins'))
            print "Imported", modules[-1]
        except ImportError as ex:
            msg.logMessage('Module could not be loaded: ' + modname)
            msg.logMessage(ex.message)

    for module in modules:
        print module.__name__
        link = pluginlink(module, placeholders)
        if link.name not in disabledatstart: link.enable()
        plugins[link.name] = link
    xglobals.plugins = plugins


def buildactivatemenu(modewidget):
    menu = QtGui.QMenu('Plugins')
    for pluginlink in plugins.values():
        if pluginlink.plugin.hidden:
            continue
        action = QtGui.QAction(pluginlink.name, menu)
        action.setCheckable(True)
        action.setChecked(pluginlink.enabled)
        action.toggled.connect(pluginlink.setEnabled)
        action.toggled.connect(modewidget.reload)
        menu.addAction(action)
    return menu


class pluginlink():
    def __init__(self, module, placeholders):
        print module
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

    def setEnabled(self, enable):
        if enable and not self.enabled:
            self.enable()
        elif self.enabled and not enable:
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
        self.setEnabled(True)
        self.instance.activate()