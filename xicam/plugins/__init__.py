from collections import OrderedDict
from PySide import QtGui
import sys
from xicam import xglobals
import importlib
from pipeline import msg
import os
import pkgutil
from xicam import safeimporter

import inspect

modules = []
plugins = OrderedDict()

disabledatstart = ['FXS', 'SPOTH5', 'Library', 'XAS','EZTest']


def initplugins(placeholders):
    import base
    global plugins, modules

    packages = pkgutil.iter_modules(__path__)
    msg.logMessage(('packages:',packages),msg.DEBUG)

    for importer, modname, ispkg in packages:

        msg.logMessage("Found plugin %s (is a package: %s)" % (modname, ispkg),msg.DEBUG)
        mod=safeimporter.import_module('.' + modname, 'xicam.plugins')
        if mod:
            modules.append(mod)
        else:
            msg.logMessage("Plugin loading aborted: "+modname,msg.CRITICAL)
            continue


    for module in modules:
        msg.logMessage(('Imported, enabling:',module.__name__),msg.DEBUG)
        for objname,obj in module.__dict__.iteritems():
            if inspect.isclass(obj):
                if issubclass(obj, base.plugin) and not obj is base.plugin and not obj is base.EZplugin:
                    link = pluginlink(module, obj, placeholders)
                    if link.name not in disabledatstart: link.enable()
                    plugins[link.name] = link
            elif type(obj) is base.EZplugin:
                link = pluginlink(module, obj, placeholders)
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
        action.toggled.connect(pluginlink.toggle)
        action.toggled.connect(modewidget.reload)
        menu.addAction(action)
    return menu


class pluginlink():
    def __init__(self, module, plugin, placeholders):
        self.plugin = plugin
        self.modulename = module.__name__
        self.module = module
        self.instance = None
        self.placeholders = placeholders
        self.name = self.plugin.name
        self._enabled = False

    def disable(self):
        del self.instance
        self.instance = None
        self._enabled = False

    def enable(self):
        #self.module = reload(sys.modules[self.modulename])
        if inspect.isclass(self.plugin):
            self.instance = self.plugin(self.placeholders)
        else:
            self.instance = self.plugin
            self.plugin.setup(self.placeholders)
        self._enabled = True

    @property
    def enabled(self):
        return self._enabled

    @enabled.setter
    def enabled(self, enabled):
        if enabled:
            self.enable()
        else:
            self.disable()

    def toggle(self):
        self.disable() if self.enabled else self.enable()

    def activate(self):
        self.enabled = True
        self.instance.activate()