# -*- coding: utf-8 -*-
"""
@author: lbluque
"""
# TODO Add viewer, external jobs and log tabs to main window
# TODO Bonus have NERSC file system look likt the local file system
# TODO Bonus have SPOT tree view get nice looking icons for datasets
# TODO change confirmationWindow to simply use QMessageBox

#from PySide import QtCore, QtGui
#from PySide.QtUiTools import QUiLoader
import os

import qdarkstyle
from PySide import QtCore, QtGui
from PySide.QtUiTools import QUiLoader

import plugins
from client import spot
from plugins.widgets import login, menu
#from threads import Worker, RunnableMethod
import threads

QtCore.Signal = QtCore.Signal
QtCore.Slot = QtCore.Slot


class SpewApp(QtGui.QMainWindow):
    """
    Class for the main window GUI.
    """

    # Signals emitted to have the worker process the appropriate function
    # loginSignal = QtCore.Signal(tuple)
    # globusLoginSignal = QtCore.Signal(str, str)
    # spotTransferSignal = QtCore.Signal(str, str)
    # globusTransferSignal = QtCore.Signal(tuple)
    # spotDownloadSignal = QtCore.Signal(tuple)
    # refreshEndpointsSignal = QtCore.Signal()

    def __init__(self, app):
        super(SpewApp, self).__init__()
        self.app = app
        with open('gui/style.stylesheet', 'r') as f:
            style = f.read()
        self.app.setStyleSheet(qdarkstyle.load_stylesheet() + style)
        # ui_file = QtCore.QFile('gui/main_window.ui')
        # ui_file.open(QtCore.QFile.ReadOnly)
        # loader = QUiLoader()
        # self.ui = loader.load(ui_file)
        # ui_file.close()
        self.ui = QUiLoader().load(QtCore.QFile('gui/main_window_v3.ui'))

        self.spotClient = None

        # Load LoginWidget
        self.loginwidget = login.LoginWidget(self)
        self.ui.loginmode.addWidget(self.loginwidget)
        self.ui.loginmode.setCurrentWidget(self.loginwidget)
        self.loginwidget.loginClicked.connect(self.handleLogin)
        self.loginwidget.logoutClicked.connect(self.handleLogout)

        # Plugins
        placeholders = [self.ui.viewmode, self.ui.rightmode, self.ui.bottommode, self.ui.toolbarmode, self.ui.leftmode]
        plugins.initplugins(placeholders)

        self.pluginmode = menu.pluginModeWidget(plugins.plugins)
        self.ui.menumode.addWidget(self.pluginmode)
        self.ui.menumode.setCurrentWidget(self.pluginmode)
        plugins.plugins['TomoViewer'].instance.activate()
        self.fileexplorer = plugins.base.fileexplorer

        # # Connect button and toolbar action signals to corresponding slots
        # self.ui.globus_login_button.clicked.connect(self.handleGlobusLogin)
        # self.ui.nersc_list_widget.itemDoubleClicked.connect(self.handleNERSCAction)
        # self.ui.nersc_back_button.clicked.connect(self.handleNERSCBack)
        # self.ui.endpoint_refresh_button.clicked.connect(self.handleEndpointRefresh)

        # Connect actions
        self.ui.actionOpen.triggered.connect(self.handleOpen)
        self.ui.actionDownload.triggered.connect(self.handleDownload)
        self.ui.actionUpload.triggered.connect(self.handleUpload)
        self.ui.actionTransfer.triggered.connect(self.handleTransfer)
        self.ui.actionDelete.triggered.connect(self.handleDelete)

        # # Start thread to handle simple gui, login actions
        # self.worker = Worker(self)
        # self.workerThread = QtCore.QThread(self, objectName='workerThread')
        # self.worker.moveToThread(self.workerThread)

        # # Connect class signals to worker
        # self.loginSignal.connect(self.worker.login)
        # self.globusLoginSignal.connect(self.worker.globusLogin)
        # self.logoutSignal.connect(self.worker.logout)
        # self.nerscActionSignal.connect(self.worker.handleNERSCAction)
        # self.nerscBackSignal.connect(self.worker.handleNERSCBack)
        # self.nerscDeleteSignal.connect(self.worker.handleNERSCDelete)
        # self.bl832dataActionSignal.connect(self.worker.handleBL832dataAction)
        # self.bl832dataBackSignal.connect(self.worker.handleBL832dataBack)
        # self.bl832dataDeleteSignal.connect(self.worker.handleBL832dataDelete)
        # self.spotMetadataSignal.connect(self.worker.handleSpotAction)
        # self.spotSearchSignal.connect(self.worker.handleSpotSearchRequest)
        # self.spotTransferSignal.connect(self.worker.handleSPOTTransfer)
        # self.globusTransferSignal.connect(self.worker.handleGlobusTransfer)
        # self.spotDownloadSignal.connect(self.worker.handleSpotDownload)
        # self.refreshEndpointsSignal.connect(self.worker.handleGlobusTableView)

        self.ui.statusbar.showMessage('Ready...')
        self.ui.closeEvent = self.closeEvent
        print 'App running on {}'.format(QtCore.QThread.currentThread().objectName())

    def closeEvent(self, event):
        r = QtGui.QMessageBox.question(self, 'Exit', 'Are you sure you want to quit SPEW',
                                       QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)
        event.ignore()
        if r == QtGui.QMessageBox.Yes:
            try:
                self.handleLogout()
            except AttributeError:
                pass
            for f in os.listdir('.tmp'):  # purge scripts in .tmp
                os.remove(os.path.join('.tmp', f))
            event.accept()

    def show(self):
        self.ui.show()
        self.ui.raise_()

    # @QtCore.Slot(QtCore.QModelIndex)
    # def onLocalItemDoubleClicked(self, index):
    #     indexItem = self.fileSystemModel.index(index.row(), 0, index.parent())
    #     filePath = self.fileSystemModel.filePath(indexItem)
    #     root = self.fileSystemModel.setRootPath(filePath)
    #     self.ui.local_tree_view.setRootIndex(root)
    #     self.ui.local_path_label.setText(filePath)
    #
    #     fname, path, page = self.getSelectedFile()
    #     if fname is not None and '.h5' in fname:
    #         self.tomoviewer.opendataset(str(filePath))
    #         #self.fillTable(self.ui.metadata_table_widget, metadata)
    #
    # def onSpotItemDoubleClicked(self):
    #     dataset, stage, page = self.getSelectedFile()
    #     if dataset is not None:
    #         self.spotMetadataSignal.emit(dataset, stage)
    #

    @QtCore.Slot(tuple)
    def handleLogin(self, credentials):
        usr, psw, systm = map(str, credentials)
        self.spotClient = spot.SpotClient(usr, psw, system=systm)
        runnable = threads.RunnableMethod(self.loginSetup, self.spotClient.login)
        self.ui.statusbar.showMessage('Logging in...', msecs=10000)
        threads.queue.put(runnable)
        threads.worker_thread.start()

    @QtCore.Slot(object)
    def loginSetup(self):
        print 'yay!'
        if self.spotClient.logged_in is True:
            self.ui.statusbar.showMessage('Loading scratch directory on {}...'.format(self.spotClient.system))
            self.fileexplorer.addNERSCTab(self.spotClient, self.spotClient.system)
            self.ui.statusbar.showMessage('Loading SPOT datasets...')
            self.fileexplorer.addSPOTTab(self.spotClient)
            self.ui.statusbar.showMessage('Ready...')
        self.loginwidget.loginSuccessful(self.spotClient.logged_in)

    def handleLogout(self):
        runnable = threads.RunnableMethod(self.loginwidget.logoutSuccessful, self.spotClient.logout)
        threads.queue.put(runnable)
        for tab in range(self.fileexplorer.count() - 1):
            self.fileexplorer.removeTab(1)

    def checkLogin(self):
        if self.spotClient is None or self.spotClient.logged_in is False:
            QtGui.QMessageBox.information(self, 'Login', 'You need to login in order to use this feature')
            return False
        else:
            return True


    # @QtCore.Slot(str)
    # def handleGlobusLogin(self, comment=''):
    #     if self.globusLogged is False:
    #         self.globusWindow = LogInWindow(title='Login to Globus.org',
    #                             label='Please login with Globus credentials.',
    #                             comment=comment)
    #
    #         if self.globusWindow.exec_() == QtGui.QDialog.Accepted:
    #             username = self.globusWindow.user_box.text()
    #             password = self.globusWindow.pass_box.text()
    #             if username == '' or password == '':
    #                 self.globusWindow.done(0)
    #                 self.handleGlobusLogin(comment='Missing username or password!')
    #             else:
    #                 self.globusLoginSignal.emit(username, password)
    #                 print ("[%s] Tasks sent" %
    #                        QtCore.QThread.currentThread().objectName()+'\n')
    #     elif self.globusLogged is True:
    #         self.printMessage('Logged in to Globus as ' +
    #                           self.globusWindow.user_box.text())
    #
    #
    def handleDelete(self):
        pass
    #     file_name, path, page = self.getSelectedFile()
    #
    #     if file_name is not None and file_name != '/':
    #         delete_comment = ('Are you sure that you want to delete ' + page +
    #                           ' file ' + file_name + ' ?')
    #         caution_comment = 'This can never be undone, NEVER.'
    #
    #         delete = self.spawnConfirmationWindow('Delete file',
    #                                               delete_comment,
    #                                               caution_comment)
    #         if delete:
    #             if page == 'Local':
    #                 if not self.fileSystemModel.remove(self.indexItem):
    #                     print 'unable to delete ' + file_name
    #                     return
    #             elif page == 'NERSC':
    #                 self.nerscDeleteSignal.emit(path + '/' + file_name)
    #                 print ("[%s] Tasks sent" %
    #                        QtCore.QThread.currentThread().objectName()+'\n')
    #             elif page == 'SPOT':
    #                 self.printMessage('How dare you try to delete datasets ' +
    #                                   'on SPOT. This event will be reported.')
    #                 return
    #             elif page == 'BL832data':
    #                 self.bl832dataDeleteSignal.emit(path + '/' + file_name)
    #                 print ("[%s] Tasks sent" %
    #                        QtCore.QThread.currentThread().objectName()+'\n')
    #
    #             print 'deleted ' + page + ' file ' + file_name
    #     else:
    #         self.printMessage('No file selected!')
    #
    def handleTransfer(self):
        if self.checkLogin() is False:
            return
    #     file_info = self.getSelectedFile()
    #
    #     if file_info[0] is None:
    #         self.printMessage('Please select a file to transfer')
    #         return
    #     elif file_info[2] == 'Local' or file_info[2] == 'NERSC':
    #         self.printMessage('Please select a file on SPOT or BL832data')
    #         return
    #
    #     comment = ('Transfer ' + file_info[2] + ' dataset ' + file_info[0] +
    #                ' to your NERSC scratch?')
    #     transfer = self.spawnConfirmationWindow('Transfer dataset', comment)
    #
    #     if transfer:
    #         if file_info[2] == 'SPOT':
    #             self.spotTransferSignal.emit(file_info[0], file_info[1])
    #             print ("[%s] Tasks sent" %
    #                    QtCore.QThread.currentThread().objectName()+'\n')
    #         if file_info[2] == 'BL832data':
    #             self.globusTransferSignal.emit(file_info)
    #
    def handleDownload(self):
        if self.checkLogin() is False:
            return
    #     file_info = self.getSelectedFile()
    #
    #     if file_info[2] == 'Local' or file_info[0] is None:
    #         self.printMessage('Please select a file on NERSC or SPOT')
    #         return
    #     elif file_info[2] == 'SPOT':
    #         fname = self.ui.spot_tree_widget.currentItem().text(0)
    #         dataset, stage = file_info[0], file_info[1]
    #     elif file_info[2] == 'NERSC':
    #         fname, path = file_info[0], file_info[1]
    #
    #     fileDialog = QtGui.QFileDialog(self, 'Save as', self.home)
    #     fileDialog.setNameFilters(['*.h5'])
    #     fileDialog.setDefaultSuffix('h5')
    #     fileDialog.selectFile(fname)
    #     fileDialog.setAcceptMode(QtGui.QFileDialog.AcceptSave)
    #
    #     if fileDialog.exec_():
    #         fpath = str(fileDialog.selectedFiles()[0])
    #         fpath, fname = os.path.split(fpath)
    #         if file_info[2] == 'SPOT':
    #             self.spotDownloadSignal.emit((dataset, stage, fpath, fname))
    #         elif file_info[2] == 'NERSC':
    #             # self.globusTransferSignal.emit()
    #             print 'Not implemented'
    #
    def handleUpload(self):
        if self.checkLogin() is False:
            return
    #     file_info = map(str, self.getSelectedFile())
    #
    #     if file_info[0] is None:
    #         self.printMessage('Please select a file to upload')
    #         return
    #     elif file_info[2] == 'SPOT' or file_info[2] == 'NERSC':
    #         self.printMessage('Please select a local file')
    #         return
    #
    #     file_size = os.path.getsize(os.path.join(file_info[1], file_info[0]))
    #     print file_info[2]
    #     print file_size
    #
    #     label = ('Upload local file ' + file_info[0] +
    #              ' to your NERSC scratch?')
    #     comment = ''
    #     if file_size > 100*(2**20):  # 100 MB:
    #         comment = ('File is larger than 100MB: Upload will be handled '
    #                    'with Globus.')
    #
    #     upload = self.spawnConfirmationWindow('Upload file', label,
    #                                           comment=comment)
    #
    #     if upload:
    #         if file_size > 100*(2**20):
    #             self.globusTransferSignal.emit(file_info)
    #         else:
    #             self.emit(QtCore.SIGNAL('handleUpload(_PyObject)'),
    #                       self.indexItem)
    #         print ("[%s] Tasks sent" %
    #                QtCore.QThread.currentThread().objectName()+'\n')
    #
    # def getSelectedFile(self):
    #     current_widget = self.ui.tabWidget.currentWidget()
    #     page = current_widget.objectName()
    #
    #     file_name = None
    #     path = None
    #
    #     if page == 'Local':
    #         file_name = self.fileSystemModel.fileName(self.indexItem)
    #         path = self.fileSystemModel.filePath(self.indexItem)
    #         path = os.path.split(str(path))[0]
    #     elif page == 'SPOT':
    #         item = self.ui.spot_tree_widget.currentItem()
    #         if item is not None:
    #             if item.childCount() == 0:
    #                 file_name = item.parent().parent().text(0) # dataset name
    #                 path = item.parent().text(0) # dataset stage
    #     elif page == 'NERSC':
    #         item = self.ui.nersc_list_widget.currentItem()
    #         if item is not None:
    #             file_name = item.text()
    #             path = self.ui.nersc_path_label.text()
    #     elif page == 'BL832data':
    #         item = self.ui.bl832data_list_widget.currentItem()
    #         if item is not None:
    #             file_name = item.text()
    #             path = self.ui.bl832data_path_label.text()
    #
    #     if page == 'Local' or page == 'NERSC' or page == 'BL832data':
    #         if file_name is not None:
    #             if len(file_name.split('.')) != 2:
    #                 file_name = None
    #
    #     return file_name, path, page
    #
    # def handleEndpointActivation(self):
    #     self.emit(QtCore.SIGNAL('handleEndpointActivation()'))
    #     self.printMessage('Not implemented yet')
    #     print ("[%s] Tasks sent" %
    #            QtCore.QThread.currentThread().objectName()+'\n')
    #
    # def handleEndpointRefresh(self):
    #     print ("[%s] Tasks sent" %
    #            QtCore.QThread.currentThread().objectName()+'\n')
    #     if self.worker.globus.logged_in is True:
    #         self.refreshEndpointsSignal.emit()
    #     else:
    #         self.printMessage('Not logged into Globus.org')
    #
    # def spawnConfirmationWindow(self, title, label='', comment=''):
    #     window = ConfirmationWindow(title=title, label=label, comment=comment)
    #     return window.exec_()
    #
    def handleRecon(self):
    #     self.printMessage('Choose reconstruction settings')
    #     filename, path, location = self.getSelectedFile()
    #     input_path, output_path = '', ''
    #
    #     if filename is not None:
    #         if location == 'SPOT':
    #             input_path = str(self.ui.spot_tree_widget.currentItem().text(0))
    #         elif filename.split('.')[1] == 'h5':
    #             input_path = os.path.join(str(path), str(filename))
    #
    #     self.reconInputs = ReconWindow(self,
    #                                    input_path=input_path,
    #                                    input_location=location)
    #
    #     if self.reconInputs.exec_():
    #         print 'running your recon duuuude!'
    #         self.printMessage('Reconstruction job submitted')
    #     else:
    #         self.printMessage('')
        print 'Recon'
    #
    # @QtCore.Slot(bool)
    # def setGlobusLogged(self, boolean):
    #         self.globusLogged = boolean
    #
    # @QtCore.Slot(object, object)
    # def fillTable(self, widgets, value):
    #     widgets.clearContents()
    #     widgets.setRowCount(len(value))
    #     widgets.setColumnWidth(0, 120)
    #
    #     if widgets.objectName() == 'globus_table_widget':
    #         widgets.setColumnCount(3)
    #         widgets.setHorizontalHeaderLabels(['Endpoint', 'Active', 'Action'])
    #     elif widgets.objectName() == 'metadata_table_widget':
    #         widgets.setColumnCount(2)
    #         widgets.setHorizontalHeaderLabels(['Field', 'Value'])
    #
    #     if type(value) is dict:
    #         for index, (key, val) in enumerate(value.iteritems()):
    #             widgets.setItem(index, 0, QtGui.QTableWidgetItem(key))
    #             widgets.setItem(index, 1, QtGui.QTableWidgetItem(str(val)))
    #             if widgets.objectName() == 'globus_table_widget':
    #                 if val is False:
    #                     button_text = 'activate'
    #                 if val is True:
    #                     button_text = 'reactivate'
    #
    #                 activate_button = QtGui.QPushButton(button_text)
    #                 activate_button.clicked.connect(self.handleEndpointActivation)
    #                 widgets.setCellWidget(index, 2, activate_button)
    #     else:
    #         child = QtGui.QTableWidgetItem()
    #         child.setText(0, unicode(value))
    #         widgets.addChild(child)
    #
    #
    def handleOpen(self):
        data_type, ok = QtGui.QInputDialog.getItem(self, 'Open Dataset', 'Data type:',
                                                   ['raw', 'reconstruction'], editable=False)
        if ok:
            fDialog = QtGui.QFileDialog(self, 'Select dataset', directory=os.path.expanduser('~'))
            fDialog.setNameFilters(['*.h5', '*.tiff', '*.tif', '*.npy'])
            fDialog.setFileMode(1)
            if fDialog.exec_():
                path = str(fDialog.selectedFiles()[0])
                self.tomoviewer.opendataset(path, data_type=data_type)


class ConfirmationWindow(QtGui.QDialog):

    def __init__(self, title='File', label='Something meaningful', comment=''):
        QtGui.QDialog.__init__(self)

        layout = QtGui.QVBoxLayout()

        self.label = QtGui.QLabel()
        self.label.setText(label)
        layout.addWidget(self.label)

        self.label2 = QtGui.QLabel()
        self.label2.setText(comment)
        layout.addWidget(self.label2)

        self.button_box = QtGui.QDialogButtonBox()
        self.button_box.setStandardButtons(QtGui.QDialogButtonBox.Yes |
                                           QtGui.QDialogButtonBox.No)
        layout.addWidget(self.button_box)

        self.setLayout(layout)

        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        self.resize(400, 60)
        self.setWindowTitle(title)


class LogInWindow(QtGui.QDialog):

    def __init__(self, title='Login', label='Please login', comment=''):
        QtGui.QDialog.__init__(self)

        layout = QtGui.QVBoxLayout()
        hlayout = QtGui.QHBoxLayout()

        self.label = QtGui.QLabel()
        self.label.setText(label)
        layout.addWidget(self.label)

        self.user_box = QtGui.QLineEdit()
        self.user_box.setPlaceholderText('Username')
        hlayout.addWidget(self.user_box)

        self.pass_box = QtGui.QLineEdit()
        self.pass_box.setPlaceholderText('Password')
        self.pass_box.setEchoMode(QtGui.QLineEdit.Password)
        hlayout.addWidget(self.pass_box)

        self.login_button = QtGui.QPushButton()
        self.login_button.setText('Login')
        hlayout.addWidget(self.login_button)

        layout.addLayout(hlayout)

        self.label2 = QtGui.QLabel()
        if comment:
            self.label2.setText(comment)
        layout.addWidget(self.label2)

        self.setLayout(layout)

        self.pass_box.returnPressed.connect(self.accept)
        self.login_button.clicked.connect(self.accept)

        self.resize(400, 60)
        self.setWindowTitle(title)
