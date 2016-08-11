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

    packages = pkgutil.iter_modules(__path__)
    msg.logMessage(('packages:',packages),msg.DEBUG)
    packages = [pkg for pkg in packages if pkg[1] not in ['widgets', 'login', 'base', 'explorer', '__init__']]
    msg.logMessage(('packages:', packages),msg.DEBUG)

    for importer, modname, ispkg in packages:
        try:
            msg.logMessage("Found plugin %s (is a package: %s)" % (modname, ispkg),msg.DEBUG)
            modules.append(importlib.import_module('.'+modname,'xicam.plugins'))
            msg.logMessage(("Imported", modules[-1]),msg.DEBUG)
        except ImportError as ex:
            msg.logMessage('Module could not be loaded: ' + modname)

            missingpackage=ex.message.replace('No module named ','')

            msgBox = QtGui.QMessageBox()
            msgBox.setText("A python package is missing! Xi-cam can try to install this for you.")
            msgBox.setInformativeText("Would you like to install "+missingpackage+"?")
            msgBox.setStandardButtons(QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)
            msgBox.setDefaultButton(QtGui.QMessageBox.Yes)

            response = msgBox.exec_()

            if response == QtGui.QMessageBox.Yes:
                import pip

                if not pip.main(['install','--user', missingpackage]):
                    msgBox = QtGui.QMessageBox()
                    msgBox.setText('Success! The missing package, '+missingpackage+', has been installed!')
                    msgBox.setInformativeText('Please restart Xi-cam now.')
                    msgBox.setStandardButtons(QtGui.QMessageBox.Ok)
                    msgBox.exec_()
                    exit(0)
                else:
                    if modname=='MOTD':
                        from xicam import debugtools
                        debugtools.frustration()
                        msgBox = QtGui.QMessageBox()
                        msgBox.setText(
                            'Sorry, ' + missingpackage + ' could not be installed. This is a Xi-cam critical library.')
                        msgBox.setInformativeText('Xi-cam cannot be loaded . Please install '+modname+' manually.')
                        msgBox.setStandardButtons(QtGui.QMessageBox.Ok)
                        response = msgBox.exec_()
                        exit(1)
                    else:
                        from xicam import debugtools
                        debugtools.frustration()
                        msgBox = QtGui.QMessageBox()
                        msgBox.setText('Sorry, '+missingpackage+' could not be installed. Try installing this package yourself, or contact the package developer.')
                        msgBox.setInformativeText('Would you like to continue loading Xi-cam?')
                        msgBox.setStandardButtons(QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)
                        response=msgBox.exec_()
                        if response==QtGui.QMessageBox.No:
                            exit(1)

            if modname == 'MOTD':
                from xicam import debugtools
                debugtools.frustration()
                msgBox = QtGui.QMessageBox()
                msgBox.setText(
                    'Sorry, ' + missingpackage + ' could not be installed. This is a Xi-cam critical library.')
                msgBox.setInformativeText('Xi-cam cannot be loaded . Please install ' + modname + ' manually.')
                msgBox.setStandardButtons(QtGui.QMessageBox.Ok)
                response = msgBox.exec_()
                exit(1)

    for module in modules:
        msg.logMessage(('Loaded:',module.__name__),msg.DEBUG)
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
       #self.module = reload(sys.modules[self.modulename])
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