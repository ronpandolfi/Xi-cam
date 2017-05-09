import numpy as np
from PySide.QtGui import *
from PySide.QtCore import *
import pyqtgraph as pg
from pyqtgraph import parametertree as pt
from pipeline import daemon



class DaemonParameter(pt.parameterTypes.GroupParameter):
    def __init__(self, openfileshandler, filter='*.edf'):
        mode = pt.Parameter.create(name='Mode', type='list', values=['Directory', 'Data Broker'])
        folder = pt.Parameter.create(name='Directory', type='str', value='', )  # TODO: make uneditable
        browse = pt.Parameter.create(name='Browse', type='action')
        filter = pt.Parameter.create(name='Filter', type='str', value=filter)
        procold = pt.Parameter.create(name='Process old', type='bool', value=True)
        self.activate = pt.Parameter.create(name='Start', type='action')
        self.handler = openfileshandler


        params = [mode, folder, browse, filter, procold, self.activate]

        super(DaemonParameter, self).__init__(name='Daemon', type='group', children=params)

        browse.sigActivated.connect(self.browse)
        self.activate.sigActivated.connect(self.toggleActivate)
        # self.activate.items.keys()[0].button.setStyleSheet('background-color:green')

    def browse(self):
        path = QFileDialog.getExistingDirectory(None, 'Set daemon directory...', '~/')
        if path is not None:
            self.param('Directory').setValue(path)

    @property
    def isActive(self):
        return self.activate.items.keys()[0].button.text() == 'Stop'

    def toggleActivate(self):
        if self.isActive:
            self.activate.items.keys()[0].button.setText('Start')
            self.activate.items.keys()[0].button.setStyleSheet('background-color:green')
        else:
            self.activate.items.keys()[0].button.setText('Stop')
            self.activate.items.keys()[0].button.setStyleSheet('background-color:red')

        self.daemon = daemon.daemon(self.param('Directory').value(),
                                    self.param('Filter').value(),
                                    self.test,procold=self.param('Process old').value())

    def test(self,*args,**kwargs):
        print args,kwargs
        self.handler(*args,**kwargs)





## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    # ######## SAVE THIS FOR DEBUGGING SEG FAULTS; issues are usually doing something outside the gui thread
    # import sys
    #
    #
    # def trace(frame, event, arg):
    #     print "%s, %s:%d" % (event, frame.f_code.co_filename, frame.f_lineno)
    #     return trace
    #
    #
    # sys.settrace(trace)

    app = QApplication([])

    ## Create window with ImageView widget
    win = QMainWindow()
    win.resize(800, 800)


    fitter = DaemonParameter()
    tree = pt.ParameterTree()
    tree.setParameters(fitter,showTop=False)
    win.setCentralWidget(tree)
    win.setWindowTitle('Fitting')
    win.show()

    QApplication.instance().exec_()
