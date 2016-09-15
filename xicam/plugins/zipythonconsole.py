from PySide import QtGui
import sys
import base
from xicam import plugins

# Overload for Py2App
def new_load_qt(api_options):
    from PySide import QtCore, QtGui, QtSvg

    return QtCore, QtGui, QtSvg, 'pyside'


from qtconsole import qt_loaders

qt_loaders.load_qt = new_load_qt

from qtconsole.rich_jupyter_widget import RichJupyterWidget
from qtconsole.inprocess import QtInProcessKernelManager

import qdarkstyle
import os



class IPythonPlugin(base.plugin):
    name = 'IPython'

    def __init__(self, *args, **kwargs):
        with open('xicam/gui/style.stylesheet', 'r') as f:
            style = f.read()
        style = (qdarkstyle.load_stylesheet() + style)

        kernel_manager = QtInProcessKernelManager()
        kernel_manager.start_kernel()
        kernel = kernel_manager.kernel
        kernel.gui = 'qt4'
        kernel.shell.push(dict(plugins.plugins))

        kernel_client = kernel_manager.client()
        kernel_client.start_channels()

        def stop():
            kernel_client.stop_channels()
            kernel_manager.shutdown_kernel()


        control = RichJupyterWidget()
        control.kernel_manager = kernel_manager
        control.kernel_client = kernel_client
        control.exit_requested.connect(stop)
        control.style_sheet = style
        control.syntax_style = u'monokai'
        control.set_default_style(colors='Linux')

        self.centerwidget = control

        self.rightwidget = None

        self.featureform = None

        self.bottomwidget = None

        self.leftwidget = None

        self.toolbar = None

        super(IPythonPlugin, self).__init__(*args, **kwargs)


