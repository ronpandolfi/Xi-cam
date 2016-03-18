# -*- coding: utf-8 -*-
"""
@author: lbluque
"""

import time
from PySide import QtGui, QtCore
from PySide.QtUiTools import QUiLoader
from client import spot
from spew import threads
QtCore.Signal = QtCore.Signal
QtCore.Slot = QtCore.Slot

NERSC_SYSTEMS = ['cori', 'edison']

class LoginDialog(QtGui.QDialog):
    """
    Class for user credential input and login to NERSC/SPOT through NEWT/SPOT API
    """

    loginClicked = QtCore.Signal(tuple)
    logoutClicked = QtCore.Signal()
    sigLoggedIn = QtCore.Signal(bool)

    def __init__(self, remotename='remote server', parent=None, fileexplorer=None):
        super(LoginDialog, self).__init__(parent)
        self.setModal(True)

        self.fileexplorer=fileexplorer

        l=QtGui.QVBoxLayout()
        l.setContentsMargins(0,0,0,0)

        self.ui = QUiLoader().load('gui/spew/login_widget.ui', self)
        l.addWidget(self.ui)
        self.ui.nersc_system_box.addItems(NERSC_SYSTEMS)
        self.ui.stackedWidget.setCurrentWidget(self.ui.login_page)
        self.ui.pass_box.setEchoMode(QtGui.QLineEdit.Password)

        self.ui.login_button.clicked.connect(self.handleLogin)
        self.ui.pass_box.returnPressed.connect(self.handleLogin)
        self.ui.logout_button.clicked.connect(self.handleLogout)
        self.setStyleSheet('background-color:#111111;')
        self.ui.nersc_system_box.setEnabled(False)

        self.setLayout(l)
        self.setWindowTitle('Login to {}'.format(remotename))

    def handleLogin(self):
        usr, pwd, systm = self.ui.user_box.text(), self.ui.pass_box.text(), self.ui.nersc_system_box.currentText()
        if usr == '':
            QtGui.QMessageBox.warning(self, 'Username missing', 'You forgot to mention who you are!')
        elif pwd == '':
            QtGui.QMessageBox.warning(self, 'Password missing',
                                      'You need to provide proof that you really are {}!'.format(usr))
        else:
            self.setCurrentWidget(self.ui.progress_page)
            self.progressMessage('Logging in...')
            self.startProgress()
            #self.loginClicked.emit((usr, pwd, systm))
            self.doLogin((usr,pwd,systm))


    def doLogin(self, credentials):
        usr, psw, systm = map(str, credentials)
        self.spotClient = spot.SpotClient(usr, psw, system=systm)
        runnable = threads.RunnableMethod(self.loginSetup, self.spotClient.login)
        threads.queue.put(runnable)
        threads.worker_thread.start()

    def loginSetup(self):
        if self.spotClient.logged_in is True:
            self.fileexplorer.addSPOTTab(self.spotClient)
#            self.ui.statusbar.showMessage('Ready...')
        self.loginSuccessful(self.spotClient.logged_in)

    def loginSuccessful(self, status):
        self.stopProgress()
        if status is True:
            #self.ui.user_label.setText('Welcome {}'.format(self.ui.user_box.text()))
            #self.setCurrentWidget(self.ui.logged_page)
            self.accept()
        else:
            self.setCurrentWidget(self.ui.login_page)
        self.ui.pass_box.clear()
        self.sigLoggedIn.emit(status)

    def setCurrentWidget(self, widget):
        self.ui.stackedWidget.setCurrentWidget(widget)

    def startProgress(self):
        self.ui.progressBar.setRange(0, 0)
        self.ui.progress_label.clear()

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