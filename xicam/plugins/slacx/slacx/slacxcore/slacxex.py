import os

from PySide import QtCore, QtUiTools

import slacxtools

@staticmethod
def throw_specific_error(msg):
    msg = 'something specific happened: ' + msg
    raise Exception(msg)

class LazyCodeError(Exception):
    def __init__(self,msg):
        super(LazyCodeError,self).__init__(self,msg)

def start_message_ui():
    ui_file = QtCore.QFile(slacxtools.rootdir+"/slacxui/message.ui")
    ui_file.open(QtCore.QFile.ReadOnly)
    msg_ui = QtUiTools.QUiLoader().load(ui_file)
    ui_file.close()
    msg_ui.setAttribute(QtCore.Qt.WA_DeleteOnClose)
    msg_ui.setWindowModality(QtCore.Qt.WindowModal)
    msg_ui.setMaximumHeight(200)
    msg_ui.message_box.setReadOnly(True)
    msg_ui.ok_button.setText('OK')
    msg_ui.ok_button.clicked.connect(msg_ui.close)
    msg_ui.ok_button.setDefault(True)
    return msg_ui

