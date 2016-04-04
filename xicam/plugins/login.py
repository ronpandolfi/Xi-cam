# -*- coding: utf-8 -*-

__author__ = "Luis Luque"
__copyright__ = "Copyright 2016, CAMERA, LBL, ALS"
__credits__ = ["Ronald J Pandolfi", "Dinesh Kumar", "Singanallur Venkatakrishnan", "Luis Luque", "Alexander Hexemer"]
__license__ = ""
__version__ = "1.2.1"
__maintainer__ = "Ronald J Pandolfi"
__email__ = "ronpandolfi@lbl.gov"
__status__ = "Beta"

"""
@author: lbluque
"""
import time
from PySide import QtGui, QtCore
from PySide.QtUiTools import QUiLoader


class LoginDialog(QtGui.QWidget):
    """
    Class for user credential input and login to NERSC/SPOT through NEWT/SPOT API
    """
    REMOTE_NAMES = ('SPOT', 'NERSC', 'Globus')

    loginClicked = QtCore.Signal(tuple)
    logoutClicked = QtCore.Signal()
    sigLoggedIn = QtCore.Signal(bool)

    def __init__(self, remotename='SPOT', parent=None):
        super(LoginDialog, self).__init__(parent)

        if remotename not in self.REMOTE_NAMES:
            raise ValueError('Remote must be one of %s, not %s' % (self.REMOTE_NAMES, remotename))

        self.ui = QUiLoader().load('gui/login_widget.ui', self)
        self.ui.stackedWidget.setCurrentWidget(self.ui.login_page)
        self.ui.pass_box.setEchoMode(QtGui.QLineEdit.Password)

        self.ui.login_button.clicked.connect(self.handleLogin)
        self.ui.pass_box.returnPressed.connect(self.handleLogin)
        self.ui.logout_button.clicked.connect(self.handleLogout)
        self.setStyleSheet('background-color:#111111;')

        # self.setWindowTitle('Login to {}'.format(remotename))

    # def show(self):
    #     self.ui.show()
    #     self.raise_()
    #     super(LoginDialog, self).show()

    # def hide(self):
    #     if self.parent() is not None:
    #         self.parent().hide()
    #     super(LoginDialog, self).hide()


    def handleLogin(self):
        usr, pwd= self.ui.user_box.text(), self.ui.pass_box.text()
        if usr == '':
            QtGui.QMessageBox.warning(self, 'Username missing', 'You forgot to mention who you are!')
        elif pwd == '':
            QtGui.QMessageBox.warning(self, 'Password missing',
                                      'You need to provide proof that you really are {}!'.format(usr))
        else:
            self.setCurrentWidget(self.ui.progress_page)
            self.progressMessage('Logging in...')
            self.startProgress()
            self.loginClicked.emit((usr, pwd))

    def loginSuccessful(self, status):
        self.stopProgress()
        if status is True:
            self.ui.user_label.setText('Welcome {}'.format(self.ui.user_box.text()))
            self.setCurrentWidget(self.ui.logged_page)
        else:
            self.setCurrentWidget(self.ui.login_page)
        self.ui.pass_box.clear()
        self.sigLoggedIn.emit(status)

    def setCurrentWidget(self, widget):
        self.ui.stackedWidget.setCurrentWidget(widget)

    def startProgress(self):
        self.ui.progressBar.setRange(0, 0)

    def stopProgress(self):
        self.ui.progressBar.setRange(0, 1)

    def progressMessage(self, message):
        # Not working for some reason
        self.ui.progress_label.setText(message)

    def handleLogout(self):
        self.setCurrentWidget(self.ui.progress_page)
        self.progressMessage('Goodbye {}...'.format(self.ui.user_box.text()))
        self.startProgress()
        self.ui.user_box.clear()
        self.logoutClicked.emit()

    def logoutSuccessful(self):
        self.stopProgress()
        self.setCurrentWidget(self.ui.login_page)