from PySide import QtGui
import sys
import base
import viewer, timeline

# Overload for Py2App
def new_load_qt(api_options):
    from PySide import QtCore, QtGui, QtSvg

    return QtCore, QtGui, QtSvg, 'pyside'


from IPython.external import qt_loaders

qt_loaders.load_qt = new_load_qt

# Necessary import for py2app build
import pygments.styles.default


from IPython.qt.console.rich_ipython_widget import RichIPythonWidget
from IPython.qt.inprocess import QtInProcessKernelManager
from IPython.lib import guisupport
import qdarkstyle
import os

from pygments import lexers

def print_process_id():
    print('Process ID is:', os.getpid())


class plugin(base.plugin):
    name = 'IPython'

    def __init__(self, *args, **kwargs):
        with open('gui/style.stylesheet', 'r') as f:
            style = f.read()
        style = (qdarkstyle.load_stylesheet() + style)

        kernel_manager = QtInProcessKernelManager()
        kernel_manager.start_kernel()
        kernel = kernel_manager.kernel
        kernel.gui = 'qt4'
        kernel.shell.push({'viewer': viewer.plugininstance, 'timeline': timeline.plugininstance})

        kernel_client = kernel_manager.client()
        kernel_client.start_channels()

        def stop():
            kernel_client.stop_channels()
            kernel_manager.shutdown_kernel()


        control = RichIPythonWidget()
        control.kernel_manager = kernel_manager
        control.kernel_client = kernel_client
        control.exit_requested.connect(stop)
        control.style_sheet = style

        self.centerwidget = control

        self.rightwidget = None

        self.leftwidget = None

        self.bottomwidget = None

        self.toolbar = None

        super(plugin, self).__init__(*args, **kwargs)


