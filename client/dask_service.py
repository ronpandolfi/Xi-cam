import client.dask_io_loop
import client.dask_local_scheduler
import client.dask_remote_scheduler
import client.dask_active_executor

from PySide import QtCore, QtGui

class ComboBoxAction(QtGui.QWidgetAction):
    def __init__(self, title, parent=None):
        QtGui.QWidgetAction.__init__(self, parent)
        pWidget = QtGui.QWidget()
        pLayout = QtGui.QHBoxLayout()
        pLabel = QtGui.QLabel(title)
        pLayout.addWidget(pLabel)

        self.pComboBox = QtGui.QComboBox()
        pLayout.addWidget(self.pComboBox)
        pWidget.setLayout(pLayout)

        self.setDefaultWidget(pWidget)

    def comboBox (self):
        return self.pComboBox

class Login(QtGui.QDialog):
    def __init__(self, machineName="", parent=None):
        super(Login, self).__init__(parent)
        self.textMachine = QtGui.QLineEdit(self)
        self.textMachine.setPlaceholderText("Machine...")
        self.textMachine.setText(machineName)
        self.textName = QtGui.QLineEdit(self)
        self.textName.setPlaceholderText("Username...")
        self.textPass = QtGui.QLineEdit(self)
        self.textPass.setPlaceholderText("Password (Empty for SSH Key)...")
        self.textPass.setEchoMode(QtGui.QLineEdit.Password)
        self.buttonLogin = QtGui.QPushButton('Login', self)
        self.buttonLogin.clicked.connect(self.handleLogin)
        layout = QtGui.QVBoxLayout(self)
        layout.addWidget(self.textMachine)
        layout.addWidget(self.textName)
        layout.addWidget(self.textPass)
        layout.addWidget(self.buttonLogin)
        self.resize(300, 100)

    def handleLogin(self):
        if len(self.textName.text()) > 0 and len(self.textMachine.text()) > 0:
            self.accept()
        else:
            self.close()

class DaskSession:
    def __init__(self, name, machine, address, exec_path, prefix_path):
        self.name = name
        self.machine = machine
        self.address = address
        self.exec_path = exec_path
        self.executor = None
        self.prefix_path = prefix_path

class DaskService(QtGui.QWidget):
    def __init__(self, *args, **kwargs):
        QtGui.QWidget.__init__(self, *args, **kwargs)

        # DASK WORKFLOW
        # TODO turn this into a class
        # convert the following into a class

        self.sessions = []
        #self.sessions.append(DaskSession("localhost", "localhost", "localhost", "", ""))
        #self.sessions.append(DaskSession("cam", "cam.lbl.gov", "cam.lbl.gov", "/home/users/hari/camera/runscript.sh", "/home/users/hari/camera/"))
        #self.sessions.append(DaskSession("Edison", "edison.nersc.gov", "edison.nersc.gov", "/global/homes/h/hkrishna/camera/runscript.sh", "/global/homes/h/hkrishna/camera/"))
        self.sessions.append(DaskSession("Bragg", "bragg.dhcp.lbl.gov", "bragg.dhcp.lbl.gov", "/opt/camera/runscript.sh", "/opt/camera/"))
        self.sessions.append(DaskSession("Cori", "cori.nersc.gov", "cori.nersc.gov", "/global/homes/h/hkrishna/camera/runscript.sh", "/global/homes/h/hkrishna/camera/"))

        #self.sessions = ["localhost", "Andromeda", "Daint", "NERSC/Edison"]
        #self.session_machines = ["localhost", "andromeda.dhcp.lbl.gov", "148.187.1.7", "edison.nersc.gov"]
        #self.session_address = ["localhost", "andromeda.dhcp.lbl.gov", "148.187.26.16", ""]
        #self.session_exec = ["", "/home/hari/runscript.sh", "/users/course79/runscript.sh",
        #                     "/usr/common/graphics/visit/camera/runscript.sh"]
        #self.executors = [None, None, None, None]

        self.actionGroup = None
        self.daskLoop = None
        self.sessionmenu = None

    def createUI(self, sessionmenu):
        from PySide import QtCore, QtGui
        self.sessionmenu = sessionmenu
        #self.comboBoxAction = ComboBoxAction("Active Session",sessionmenu)
        self.actionGroup = QtGui.QActionGroup(sessionmenu)
        for session in self.sessions:
            action = QtGui.QAction(session.name, sessionmenu, checkable=True)
            if session.name == "localhost":
                action.setChecked(True)
            action.triggered.connect(self.activeSessionChanged)
            self.actionGroup.addAction(action)
            sessionmenu.addAction(action)

        #self.comboBoxAction.comboBox().activated.connect(self.activeSessionChanged)
        #sessionmenu.addAction(self.comboBoxAction)

    def start(self):
        self.daskLoop = client.dask_io_loop.DaskLoop()
        # for now return and don't create localhost
        if True:
           return
        try:
            # create a local active executor
            local_scheduler = client.dask_local_scheduler.LocalScheduler(self.daskLoop)
            local_scheduler.execute()

            self.sessions[0].executor = local_scheduler
            self.sessionmenu.setTitle("Active Session (localhost)")
            client.dask_active_executor.active_executor = self.sessions[0]
        except:
            print("Issues connecting to localhost")

    def closeAllConnections(self):
        #msg.logMessage("Closing all connections")

        # stop any existing executors
        for e in range(len(self.sessions)):
            if self.sessions[e].executor is not None:
                self.sessions[e].executor.close()

        self.daskLoop.loop.stop()
        self.daskLoop.loop.close()

        self.daskLoop.loop.instance().add_callback(self.daskLoop.loop.instance().stop)

    def activeSessionChanged(self):
        # w = self
        obj = 0
        for (i, ac) in enumerate(self.actionGroup.actions()):
            if self.sender().text() == ac.text():
                obj = i
                break

        if self.sessions[obj].executor != None:
            client.dask_active_executor.active_executor = self.sessions[obj]
            self.sessionMenu.setText("Active Session ({0})".format(self.session[obj].machine))
        else:
            # setup connection
            login = Login(self.sessions[obj].machine)
            if login.exec_() == QtGui.QDialog.Accepted:
                username = str(login.textName.text())
                machine = str(login.textMachine.text())
                password = str(login.textPass.text())

                #msg.logMessage((username, machine),msg.DEBUG)  # , password
                self.sessions[obj].executor = client.dask_remote_scheduler.RemoteScheduler(machine, username, self.daskLoop, password, self.sessions[obj].address, self.sessions[obj].exec_path)
                self.sessionmenu.setTitle("Active Session ({0})".format(self.sessions[obj].machine))
       
                import time
                time.sleep(5)
                self.sessions[obj].executor.execute()
                client.dask_active_executor.active_executor = self.sessions[obj]
                msg = QtGui.QMessageBox()
                msg.setIcon(QtGui.QMessageBox.Information)
                msg.setText("Connection to {0} complete.".format(self.sessions[obj].machine))
                msg.exec_()

