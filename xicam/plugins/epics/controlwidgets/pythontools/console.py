from PySide.QtGui import *




def new_load_qt(api_options):
    from PySide import QtCore, QtGui, QtSvg

    return QtCore, QtGui, QtSvg, 'pyside'


from qtconsole import qt_loaders

qt_loaders.load_qt = new_load_qt

from qtconsole.rich_jupyter_widget import RichJupyterWidget
from qtconsole.inprocess import QtInProcessKernelManager

import qdarkstyle





class ipythonconsole(RichJupyterWidget):

    def __init__(self):
        super(ipythonconsole, self).__init__()
        self.kernel_manager = QtInProcessKernelManager()
        self.kernel_manager.start_kernel()
        self.kernel = self.kernel_manager.kernel
        self.kernel.gui = 'qt4'

        self.kernel_client = self.kernel_manager.client()
        self.kernel_client.start_channels()

        self.exit_requested.connect(self.stop)
        self.style_sheet = (qdarkstyle.load_stylesheet())
        self.syntax_style = u'monokai'
        self.set_default_style(colors='Linux')

    def stop(self):
        self.kernel_client.stop_channels()
        self.kernel_manager.shutdown_kernel()

    def push(self,**kwargs):
        self.kernel.shell.push(kwargs)