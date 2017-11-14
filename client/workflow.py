from PySide import QtGui
import camlink
from camlink.services import graph as task_graph
from camlink.comm.machine_profile import global_mp

from threading import Thread

import dask
import dask.distributed

class ComboBoxAction(QtGui.QWidgetAction):
    def __init__(self, title, parent=None):
        QtGui.QWidgetAction.__init__(self, parent)
        pWidget = QtGui.QWidget()
        pLayout = QtGui.QHBoxLayout()
        pLabel = QtGui.QLabel(title)
        pLayout.addWidget(pLabel)

        pComboBox = QtGui.QComboBox()
        pLayout.addWidget(pComboBox)
        pWidget.setLayout(pLayout)

        self.setDefaultWidget(pWidget)

        # def comboBox (self):
        #    return self.pComboBox


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


class XicamWorkflow(QtGui.QWidget):
   def __init__(self):
        QtGui.QWidget.__init__(self)
        self.graph = task_graph.Graph()
        global_mp.read_machine_list("/Users/hari/master_connections.txt")
        self.actionGroup = None

        self.graph = None
        self.client = None
        self.connection = None

        self.template = [
            {
                "machine": "localhost",
                "apps": 
                    [
                        {
                            "name": "dask/dask-scheduler"
                        }
                    ],
            }
        ]

   def run_paws(self, paws):
        import os
    
        if self.client is None:
            return

        file_list = paws.get_input_setting('batch', 'file_list')
        new_file_list = []

        for file in file_list:
              new_file = "/tmp/" + os.path.basename(file) 
              new_file_list.append(new_file)
              print file, new_file
              self.graph[self.connection].node.put(file, new_file)

        paws.set_input('batch', 'file_list', new_file_list)
        paws.save_to_wfl("/tmp/mno.wfl")
        contents = open("/tmp/mno.wfl").read()

        if self.client is None:
            return

        def paws_wrapper(papi):
            import paws
            import paws.api
            from collections import OrderedDict
         
            f = open("/tmp/abc.wfl","w")
            f.write(papi)
            f.close()

            paw = paws.api.start()
            paw.load_from_wfl("/tmp/abc.wfl")
            paw.execute()
            #paw.save_to_wfl("/tmp/xyz.wfl")

            _wfname = "img_process"

            result = {}

            op_tag = 'Read Image'
            key = 'image_data'
            result[op_tag+":"+key] = paw.get_output(op_tag,key,_wfname)

            op_tag = 'Integrate to 2d'
            key = 'I_at_q_chi'
            result[op_tag+":"+key] = paw.get_output(op_tag,key,_wfname)

            op_tag = 'Integrate to 1d'
            key = 'q_I'
            result[op_tag+":"+key] = paw.get_output(op_tag,key,_wfname)

            op_tag = 'log(I) 1d'
            key = 'x_logy'
            result[op_tag+":"+key] = paw.get_output(op_tag,key,_wfname)

            op_tag = 'log(I) 2d'
            key = 'logx'
            result[op_tag+":"+key] = paw.get_output(op_tag,key,_wfname)

            return result

        res = self.client.submit(paws_wrapper, contents)
        print res.result()

        #for keys in res.keys():
        #    key = keys.split(":")
        #    paws.set_input("img_process", key[0], res[keys])
        return res.result()
       
   def setup(self, menubar):
        sessionmenu = QtGui.QMenu('Sessions')
        actionGroup = QtGui.QActionGroup(sessionmenu)
        self.actionGroup = actionGroup

        for name in global_mp.config.keys():
           
            action = QtGui.QAction(name, sessionmenu, checkable=True)
            if name == "localhost":
                action.setChecked(True)
            action.triggered.connect(self.activesessionchanged)
            actionGroup.addAction(action)
            sessionmenu.addAction(action)

        menubar.addMenu(sessionmenu)

        # self.daskLoop = client.dask_io_loop.DaskLoop()
        # try:
        #     # create a local active executor
        #     local_scheduler = client.dask_local_scheduler.LocalScheduler(self.daskLoop)
        #     local_scheduler.execute()
        #     self.executors[0] = local_scheduler
        #     self.sessionmenu.setTitle("Active Session (localhost)")
        #     client.dask_active_executor.active_executor = local_scheduler
        # except:
        #     msg.logMessage("Issues connecting to localhost",msg.ERROR)

   def closeAllConnections(self):
        msg.logMessage("Closing all connections")

        # stop any existing executors
        # for e in range(len(self.executors)):
        #     if self.executors[e] is not None:
        #         self.executors[e].close()
        # self.daskLoop.loop.stop()
        # self.daskLoop.loop.close()

        # self.daskLoop.loop.instance().add_callback(self.daskLoop.loop.instance().stop)

   def activesessionchanged(self):
        import copy
        obj = ""
        for ac in self.actionGroup.actions():
            if self.sender().text() == ac.text():
                obj = str(ac.text())
                break

        try:
            if self.connection is not None:
                # close previous link..
                # todo turn this into a function...
                #self.graph[self.connection].service.stream.disconnect()
                #self.graph[self.connection].service.stream.close()
                self.graph[self.connection].node.client.close()

            machine_list = copy.deepcopy(self.template)
            machine_list[0]["machine"] = obj
            print machine_list

            connection = obj
            graph = task_graph.Graph()
            result = graph.start_services(machine_list)

            if result is True:
                print("Services started successfully to: ", machine_list)

            result = graph.setup()

            if result is True:
                print("Services setup successfully to: ", machine_list)

            for machine in graph.machines:
                for service in machine.all_services().values():
                    print("Available: ", machine.name, service.name(), service.input_vars(), service.output_vars())

            graph[obj].configure_tasks(["dask-cluster"])
            graph.start_tasks()
            graph.connect()
            graph.execute()

            port = graph[obj]["dask-cluster"].request_meta_data()[0][0]

            if obj != "localhost":
                fp = graph[obj].node.get_free_local_port()
                graph[obj].node.forward_tunnel(fp, "localhost", port)
                port = fp

            client = dask.distributed.Client("tcp://localhost:" + str(port))
            self.connection = connection
            self.client = client
            self.graph = graph
        except Exception as e:
            print("Exception : ", repr(e))
        # if self.executors[obj] != None:
        #     client.dask_active_executor.active_executor = self.executors[obj]
        #     self.sessionMenu.setText("Active Session ({0})".format(self.session_machines[obj]))
        # else:
        #     # setup connection
        #     login = Login(self.session_machines[obj])
        #     if login.exec_() == QtGui.QDialog.Accepted:
        #         username = str(login.textName.text())
        #         machine = str(login.textMachine.text())
        #         password = str(login.textPass.text())
        #         msg.logMessage((username, machine),msg.DEBUG)  # , password
        #         self.executors[obj] = client.dask_remote_scheduler.RemoteScheduler(machine, username, self.daskLoop,
        #                                                                            password, self.session_address[obj],
        #                                                                            self.session_exec[obj])
        #         self.sessionmenu.setTitle("Active Session ({0})".format(self.session_machines[obj]))
        #
        #         import time
        #         time.sleep(5)
        #         self.executors[obj].execute()
        #         client.dask_active_executor.active_executor = self.executors[obj]

