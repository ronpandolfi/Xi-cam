#import os
import glob
import traceback
from collections import Iterator
from datetime import datetime as dt

from PySide import QtCore, QtUiTools
from PySide import QtCore
from operations.slacxop import Operation

# TODO: Make scratch directory and other cfg'ables into a cfg file

version='0.1.0'

qdir = QtCore.QDir(__file__)
qdir.cdUp()
qdir.cdUp()
rootdir = qdir.path() 
#rootdir = os.path.split( qdir.absolutePath() )[0]
print 'slacxtools.rootdir: {}'.format(rootdir)
qdir.cdUp()
qdir.cd('scratch')
scratchdir = qdir.path()
#scratchdir = os.path.split( qdir.absolutePath() )[0]
print 'slacxtools.scratchdir: {}'.format(scratchdir)

class LazyCodeError(Exception):
    def __init__(self,msg):
        super(LazyCodeError,self).__init__(self,msg)

class WfWorker(QtCore.QObject):
    """
    Container for storing and executing parts of a workflow,
    to be pushed onto QtCore.QThread(s) as needed.
    """
    
    finished = QtCore.Signal()
    opDone = QtCore.Signal(str,Operation)

    def __init__(self,to_run=None,parent=None):
        super(WfWorker,self).__init__(parent)
        self.to_run = to_run

    def work(self):
        try:
            for itm in self.to_run:
                # run and update the Operation in this TreeItem
                op = itm.data
                #op.run_and_update()
                op.run()
                self.opDone.emit(itm.tag(),op)
                #self.wfman.run_and_update(item)
            self.thread().quit()
        except Exception as ex:
            # TODO: Handle this exception from wfman's pov
            tb = traceback.format_exc()
            msg = str('Error encountered during execution. \n'
                + 'Error message: {} \n'.format(ex.message) 
                + 'Stack trace: {}'.format(tb)) 
            print msg
            self.thread().quit()
            raise ex

class FileSystemIterator(Iterator):

    def __init__(self,dirpath,regex):
        self.paths_done = []
        self.dirpath = dirpath
        self.rx = regex
        super(FileSystemIterator,self).__init__()

    def next(self):
        #import pdb; pdb.set_trace()
        batch_list = glob.glob(self.dirpath+'/'+self.rx)
        for path in batch_list:
            if not path in self.paths_done:
                self.paths_done.append(path)
                return [path]
        return [None]

def throw_specific_error(msg):
    msg = 'something specific happened: ' + msg
    raise Exception(msg)

def start_message_ui():
    ui_file = QtCore.QFile(rootdir+"/slacxui/message.ui")
    ui_file.open(QtCore.QFile.ReadOnly)
    msg_ui = QtUiTools.QUiLoader().load(ui_file)
    ui_file.close()
    msg_ui.setAttribute(QtCore.Qt.WA_DeleteOnClose)
    msg_ui.setWindowModality(QtCore.Qt.WindowModal)
    #msg_ui.setMaximumHeight(200)
    msg_ui.message_box.setReadOnly(True)
    msg_ui.ok_button.setText('OK')
    msg_ui.ok_button.clicked.connect(msg_ui.close)
    msg_ui.ok_button.clicked.connect(msg_ui.deleteLater)
    msg_ui.ok_button.setDefault(True)
    return msg_ui

def dtstr():
    """Return date and time as a string"""
    return dt.strftime(dt.now(),'%Y %m %d, %H:%M:%S')

def timestr():
    """Return time as a string"""
    return dt.strftime(dt.now(),'%H:%M:%S')

#class OpExecThread(QtCore.QThread):
#    """Thread subclass for executing an Operation"""
#
#    def __init__(self,op,parent=None):
#        super(OpExecThread,self).__init__(parent)
#        self.op = op 
#
#    def run(self):
#        """Calling QThread.start() is expected to cause this run() method to run"""
#        self.op.run()
#        # Start event handler:
#        self.exec_()

