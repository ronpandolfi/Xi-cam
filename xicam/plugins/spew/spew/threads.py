# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 17:22:00 2015

@author: lbluque
"""

# TODO implement logging with error tracebacks and thread prints, etc

import os
import Queue
import multiprocessing as mp
from PySide import QtCore
from client.globus import GlobusClient, GLOBUSError
from client.spot import SpotClient
from client.user import AUTHError
import time


QtCore.Signal = QtCore.Signal
QtCore.Slot = QtCore.Slot


class Emitter(QtCore.QObject):
    """
    Class that holds signals that can be emitted by a QRunnable
    """

    sigRetValue = QtCore.Signal(object)

    def __init__(self):
        super(Emitter, self).__init__()


class RunnableMethod(QtCore.QRunnable):
    """
    Runnable that will execute a given method from a QThreadPool and emit the response to the given callback function
    """

    def __init__(self, callback_slot, method, *args, **kwargs):
        super(RunnableMethod, self).__init__()
        self.emitter = Emitter()
        self._callback_slot = callback_slot
        self._method = method
        self.args = args
        self.kwargs = kwargs

        self.emitter.sigRetValue.connect(self._callback_slot,  QtCore.Qt.QueuedConnection)

    def run(self):
        print 'Started {0} in thread {1}, will emit back to {2}'.format(self._method.__name__,
                                                                        QtCore.QThread.currentThread(),
                                                                        self._callback_slot.__name__)
        # self.emitter.sigFinished.connect(self._callback_slot)  # Connect here or in constructor?
        try:
            value = self._method(*self.args, **self.kwargs)
            print value
            if value is None:
                value=False
            self.emitter.sigRetValue.emit(value)
        except Exception, e:
            raise e
        #     value = e
        # finally:
        #     self.emitter.sigFinished.emit(value)


class RunnableIterator(RunnableMethod):
    """
    Runnable that will loop through an iterator and emit a signal representing the progress of a generator method
    """

    def __init__(self, callback_slot, generator, *args, **kwargs):
        super(RunnableIterator, self).__init__(callback_slot, generator, *args, **kwargs)

    def run(self):
        print 'Started {0} in thread {1}, will update to {2}'.format(self._method.__name__,
                                                                     QtCore.QThread.currentThread(),
                                                                     self._callback_slot.__name__)
        for status in self._method(*self.args, **self.kwargs):
            self.emitter.sigRetValue.emit(status)


class Worker(QtCore.QObject):
    """
    Daemon worker that contains a Queue and QThreadPool for running jobs
    """

    def __init__(self, queue, parent=None):
        super(Worker, self).__init__(parent)
        self.queue = queue
        self.pool = QtCore.QThreadPool(self)  # Should I use globalInstance()?
        self.pool.setMaxThreadCount(mp.cpu_count())

    def __del__(self):
        self.queue.join()

    def startRunnable(self, runnable, priority=0):
        self.pool.start(runnable, priority)

    def run(self):
        while True:
            item = self.queue.get()
            print "Worker got item {} off queue".format(type(item))
            self.startRunnable(item)
            self.queue.task_done()
            time.sleep(.3)

# Application globals

queue = Queue.Queue()
worker = Worker(queue)
worker_thread = QtCore.QThread(objectName='workerThread')
worker.moveToThread(worker_thread)
worker_thread.started.connect(worker.run)


class Worker1(QtCore.QObject):
    """
    Class for running simple GUI actions, logging in and SPOT dataset transfer
    actions from a separate thread
    """
    taskStart = QtCore.Signal()
    taskDone = QtCore.Signal()
    workingMessage = QtCore.Signal(str)
    setTextSignal = QtCore.Signal(object, str)

    clearLOCALSignal = QtCore.Signal()
    clearSCRATCHSignal = QtCore.Signal()
    clearSPOTSignal = QtCore.Signal()
    clearNERSCPathSignal = QtCore.Signal()
    clearBL832dataSignal = QtCore.Signal()
    clearBL832pathSignal = QtCore.Signal()
    clearEndpointsSignal = QtCore.Signal()
    globusLoggedSignal = QtCore.Signal(bool)
    globusLoginSignal = QtCore.Signal(str)
    fillTableSignal = QtCore.Signal(object, object)
    fillTreeSignal = QtCore.Signal(object, object)
    fillLocalSignal = QtCore.Signal()
    fillListSignal = QtCore.Signal(object, object)

    loginResultSignal = QtCore.Signal(bool)
    logoutFinishedSignal = QtCore.Signal()
    downloadSignal = QtCore.Signal(int)
    percentSignal = QtCore.Signal(int)

    def __init__(self, parent):
        QtCore.QObject.__init__(self)
        self.parent = parent
        self.spot = None
        self.nersc_system = None
        self.globus = None
        self.nersc_directory = ''
        self.bl832data_directory = ''

    def __del__(self):
        self.wait()

    @QtCore.Slot(object)
    def start(self):
        # self.workingMessage.connect(self.parent.printMessage)
        # self.taskDone.connect(self.parent.onFinished)
        # self.taskStart.connect(self.parent.onStart)
        #
        # self.taskDone.connect(self.parent.onFinished)
        # self.taskStart.connect(self.parent.onStart)
        # self.percentSignal.connect(self.parent.onProgress)
        # self.workingMessage.connect(self.parent.printMessage)
        #
        # self.downloadSignal.connect(self.parent.setDownloadRange)
        # self.percentSignal.connect(self.parent.onProgress)
        # self.loginResultSignal.connect(self.parent.handleLoginGui)
        # self.logoutFinishedSignal.connect(self.parent.handleLogoutGui)
        #
        # self.fillTreeSignal.connect(self.parent.fillTree)
        # self.fillTableSignal.connect(self.parent.fillTable)
        # self.fillListSignal.connect(self.parent.fillList)
        # self.globusLoginSignal.connect(self.parent.handleGlobusLogin)
        # self.globusLoggedSignal.connect(self.parent.setGlobusLogged)
        # self.clearLOCALSignal.connect(self.parent.ui.local_tree_view.reset)
        # self.clearSCRATCHSignal.connect(self.parent.ui.nersc_list_widget.clear)
        # self.clearSPOTSignal.connect(self.parent.ui.spot_tree_widget.clear)
        # self.clearNERSCPathSignal.connect(self.parent.ui.nersc_path_label.clear)
        # self.clearBL832dataSignal.connect(self.parent.ui.bl832data_list_widget.clear)
        # self.clearBL832dataSignal.connect(self.parent.ui.bl832data_path_label.clear)
        # self.clearEndpointsSignal.connect(self.parent.ui.globus_table_widget.clearContents)
        # self.setTextSignal.connect(self.parent.setText)

        print '[%s] start()' % QtCore.QThread.currentThread().objectName()+'\n'

    @QtCore.Slot(tuple)
    def login(self, login_creds):
        print ("[%s] Processing handleLogin" %
               QtCore.QThread.currentThread().objectName())
        username, password, self.nersc_system = map(str, login_creds)
        self.taskStart.emit()
        self.workingMessage.emit("Logging In...")

        if username != '' and password != '':
            self.spot = SpotClient(username, password, system=self.nersc_system)

            if self.spot.logged_in:
                print("Success!")
                self.handleNERSCDirectoryList()
                self.handleSpotSearchRequest({'end_station': 'bl832'})
                self.workingMessage.emit("Logged in succesfully")
                self.globusLoginSignal.emit('Required for dataset transfers' +
                                            ' from local and bl832data')
            else:
                self.workingMessage.emit("Wrong Username or Password!")

            self.loginResultSignal.emit(self.spot.logged_in)

        else:
            self.workingMessage.emit("Username or password missing!")
            self.loginResultSignal.emit(False)

        self.taskDone.emit()

    @QtCore.Slot(str, str)
    def globusLogin(self, username, password):
        self.taskStart.emit()
        self.workingMessage.emit("Logging in to Globus.org...")

        print ("[%s] Processing handleGlobusLogin" %
               QtCore.QThread.currentThread().objectName())

        username, password = str(username), str(password)

        self.globus = GlobusClient(username, password)

        if self.globus.logged_in:
            user_endpoints = self.globus.find_user_endpoints()

            for endpoint in user_endpoints:
                self.globus.add_endpoint(endpoint)

            self.local_endpoint = self.globus.determine_local_endpoint()
            if self.local_endpoint is None:
                self.setTextSignal.emit(self.parent.ui.local_endpoint_label,
                                        'Local endpoint not determined')
            else:
                self.setTextSignal.emit(self.parent.ui.local_endpoint_label,
                                        'Local: ' + self.local_endpoint)
            root_path = '//data/scratch' #'//data/scratch/' + username
            try:
                dir_contents = self.globus.get_dir_contents('alsuser#bl832data',
                                                            root_path)
                self.fillListSignal.emit(self.parent.ui.bl832data_list_widget,
                                         dir_contents)
                self.setTextSignal.emit(self.parent.ui.bl832data_path_label,
                                        root_path)
                self.bl832data_directory = root_path
            except GLOBUSError:
                self.setTextSignal.emit(self.parent.ui.bl832data_path_label,
                                        'alsuser#bl832data is not active')

            self.handleGlobusTableView()
            self.globusLoggedSignal.emit(True)
            self.workingMessage.emit("Logged in succesfully")

        else:
            self.workingMessage.emit("Wrong Username or Password!")

        self.taskDone.emit()

    def logout(self):
        self.taskStart.emit()
        print ("[%s] Processing handleLogout" %
               QtCore.QThread.currentThread().objectName())

        self.workingMessage.emit("Logging Out...")

        if self.spot is not None:
            self.spot.logout()

        if self.globus is not None:
            self.globus.logout()

        self.globusLoggedSignal.emit(False)
        self.logoutFinishedSignal.emit()
        self.taskDone.emit()

    def handleNERSCDirectoryList(self):
        self.taskStart.emit()
        print ("[%s] Processing handleNERSCDirectoryList" %
               QtCore.QThread.currentThread().objectName())
        self.workingMessage.emit("Loading NERSC scratch directory")

        if self.spot.scratch_dir is None:
            self.spot.get_scratch_dir()
            if self.spot.scratch_dir is None:
                self.workingMessage.emit("NERSC scratch directory not found!")
                self.setTextSignal.emit(self.parent.ui.nersc_path_label,
                                        'Scratch directory not found')
                self.taskDone.emit()
                return

        dir_contents = self.spot.get_dir_contents(self.spot.scratch_dir,
                                                  system=self.nersc_system)
        self.fillListSignal.emit(self.parent.ui.nersc_list_widget,
                                 dir_contents)
        self.setTextSignal.emit(self.parent.ui.nersc_path_label,
                                self.spot.scratch_dir)
        self.nersc_directory = self.spot.scratch_dir

        self.workingMessage.emit("NERSC scratch directory found!")
        self.taskDone.emit()

    def handleGlobusTableView(self):
        endpoint_view = {}
        for endpoint in self.globus.endpoints:
            self.globus.add_endpoint(endpoint)
            endpoint_view[endpoint] = self.globus.endpoints[endpoint]['activated']
        self.fillTableSignal.emit(self.parent.ui.globus_table_widget,
                                  endpoint_view)
        self.workingMessage.emit('Finished updating endpoints')

    @QtCore.Slot(str)
    def handleNERSCAction(self, filename):
        self.taskStart.emit()
        print ("[%s] Processing handleSCRATCHAction" %
               QtCore.QThread.currentThread().objectName())

        if len(filename.split('.')) == 1:
            self.nersc_directory += '/' + str(filename)
            self.workingMessage.emit("Loading NERSC directory")
            dir_contents = self.spot.get_dir_contents(self.nersc_directory,
                                                      system=self.nersc_system)
            self.fillListSignal.emit(self.parent.ui.nersc_list_widget,
                                     dir_contents)
            self.setTextSignal.emit(self.parent.ui.nersc_path_label,
                                    self.nersc_directory)
            self.workingMessage.emit("NERSC directory loaded!")
        elif ".h5" in filename:
            print filename + " is an h5 file!"
        self.taskDone.emit()

    def handleNERSCBack(self):
        self.taskStart.emit()
        print ("[%s] Processing handleNERSCBack" %
               QtCore.QThread.currentThread().objectName())
        self.workingMessage.emit('Loading NERSC directory')
        self.nersc_directory = self.nersc_directory[:self.nersc_directory.rindex('/')]
        dir_contents = self.spot.get_dir_contents(self.nersc_directory,
                                                  system=self.nersc_system)
        self.fillListSignal.emit(self.parent.ui.nersc_list_widget,
                                 dir_contents)
        self.setTextSignal.emit(self.parent.ui.nersc_path_label,
                                self.nersc_directory)
        self.workingMessage.emit("NERSC directory loaded!")
        self.taskDone.emit()

    @QtCore.Slot(str)
    def handleBL832dataAction(self, filename):
        self.taskStart.emit()
        print ("[%s] Processing handleBL832dataAction" %
               QtCore.QThread.currentThread().objectName())

        if len(filename.split('.')) == 1:
            self.bl832data_directory += '/' + str(filename)
            self.workingMessage.emit("Loading BL832data directory")
            try:
                dir_contents = self.globus.get_dir_contents('alsuser#bl832data',
                                                            self.bl832data_directory)
                self.fillListSignal.emit(self.parent.ui.bl832data_list_widget,
                                         dir_contents)
                self.setTextSignal.emit(self.parent.ui.bl832data_path_label,
                                        self.bl832data_directory)

            except GLOBUSError:
                self.setTextSignal.emit(self.parent.ui.bl832data_path_label,
                                        'alsuser#bl832data is not active')
            self.workingMessage.emit("BL832data directory loaded!")
        elif ".h5" in filename:
            print filename + " is an h5 file!"
        self.taskDone.emit()

    def handleBL832dataBack(self):
        self.taskStart.emit()
        print ("[%s] Processing handleBL832dataBack" %
               QtCore.QThread.currentThread().objectName())

        self.workingMessage.emit("Loading BL832data directory")
        self.globus.add_endpoint('alsuser#bl832data')
        self.bl832data_directory = self.bl832data_directory[:self.bl832data_directory.rindex('/')]
        #if self.bl832data_directory == '' or self.bl832data_directory == '/':
        #    self.bl832data_directory = '/~'

        try:
            dir_contents = self.globus.get_dir_contents('alsuser#bl832data',
                                                        self.bl832data_directory)
            self.fillListSignal.emit(self.parent.ui.bl832data_list_widget,
                                     dir_contents)
            self.setTextSignal.emit(self.parent.ui.bl832data_path_label,
                                    self.bl832data_directory)
            self.workingMessage.emit("BL832data directory loaded!")
        except GLOBUSError:
            self.setTextSignal.emit(self.parent.ui.bl832data_path_label,
                                    'alsuser#bl832data is not active')

        self.taskDone.emit()

    @QtCore.Slot(str, str)
    def handleSpotAction(self, dataset, stage):
        self.taskStart.emit()
        self.workingMessage.emit('Fetching dataset metadata')
        metadata = self.spot.get_dataset_attributes(str(dataset), str(stage))
        self.fillTableSignal.emit(self.parent.ui.metadata_table_widget,
                                  metadata)
        self.taskDone.emit()

    @QtCore.Slot(dict)
    def handleSpotSearchRequest(self, search_params):
        self.taskStart.emit()
        print ("[%s] Processing handleSpotSearchRequest" %
               QtCore.QThread.currentThread().objectName())
        self.workingMessage.emit("Searching for SPOT datasets")

        results = self.spot.search(**search_params)

        if results == "[]":
            self.clearSPOTSignal.emit()
            self.taskDone.emit()
            self.workingMessage.emit("No results from SPOT Search!")
            return

        self.handleSPOTTreeView(results)
        self.workingMessage.emit("SPOT Search finished!")
        self.taskDone.emit()

    def handleSPOTTreeView(self, spot_data):
        self.taskStart.emit()
        print ("[%s] Processing handleSPOTTreeView" %
               QtCore.QThread.currentThread().objectName())

        tree_data = {}
        for index in range(len(spot_data)):
            derived_data = {spot_data[index]['fs']['stage']:
                            spot_data[index]['name']}
            if 'derivatives' in spot_data[index]['fs'].keys():
                derivatives = spot_data[index]['fs']['derivatives']
                for d_index in range(len(derivatives)):
                    stage = derivatives[d_index]['dstage']
                    name = derivatives[d_index]['did'].split('/')[-1]
                    derived_data[stage] = name

            dataset = spot_data[index]['fs']['dataset']
            tree_data[dataset] = derived_data  # derived_list[index]
            self.fillTreeSignal.emit(self.parent.ui.spot_tree_widget,
                                     tree_data)

        self.taskDone.emit()
        self.workingMessage.emit("SPOT datasets loaded!")

    @QtCore.Slot(str)
    def handleNERSCDelete(self, file_path):
        self.taskStart.emit()

        print "Deleting file from NERSC!"
        try:
            self.spot.check_login()
            self.spot.delete_file(str(file_path), 'cori')
            fname = file_path.split('/')[-1]
            self.workingMessage.emit('Deleted {0} from NERSC'.format(fname))
        except AUTHError:
            self.workingMessage.emit('Not logged in to NERSC')
            return

        dir_contents = self.spot.get_dir_contents(self.nersc_directory,
                                                  system=self.nersc_system)
        self.fillListSignal.emit(self.parent.ui.nersc_list_widget,
                                 dir_contents)
        self.taskDone.emit()

    @QtCore.Slot(str)
    def handleBL832dataDelete(self, file_name):
        self.taskStart.emit()
        print ("[%s] Processing handleBL832dataDelete" %
               QtCore.QThread.currentThread().objectName())

        try:
            self.globus.check_login()
            self.globus.delete_file('alsuser#bl832data', str(file_name))
            self.workingMessage.emit("Deleting file from BL832data!")
            self.workingMessage.emit('Deleted {0} from BL832data'.format(file_name))
        except AUTHError:
            self.workingMessage.emit('Not logged in to Globus')
            return
        except:
            self.workingMessage.emit('Error submitting Globus delete request')
            raise

        dir_contents = self.globus.get_dir_contents('alsuser#bl832data',
                                                    self.bl832data_directory)
        self.fillListSignal.emit(self.parent.ui.bl832data_list_widget,
                                 dir_contents)
        self.taskDone.emit()

    @QtCore.Slot(str, str)
    def handleSPOTTransfer(self, dataset, stage):
        self.taskStart.emit()
        print "Transfering dataset to SCRATCH!"
        self.workingMessage.emit("Transfering SPOT dataset to NERSC scratch")

        full_path = str(self.spot.get_file_location(str(dataset), str(stage)))
        self.spot.copy_file(full_path, self.spot.scratch_dir, system=self.nersc_system)
        dir_contents = self.spot.get_dir_contents(self.nersc_directory,
                                                  system=self.nersc_system)
        self.fillListSignal.emit(self.parent.ui.nersc_list_widget,
                                 dir_contents)

        self.workingMessage.emit("Finished transfering dataset!")
        self.taskDone.emit()

    @QtCore.Slot(tuple)
    def handleGlobusTransfer(self, file_info):
        self.taskStart.emit()
        file_name, path, page = map(str, file_info)
        print ("[%s] Processing handleGlobusTransfer" %
               QtCore.QThread.currentThread().objectName())
        src_path = os.join(path, file_name)
        if page == 'BL832data':
            src_endpoint = 'alsuser#bl832data'
        elif page == 'Local':
            src_endpoint = self.local_endpoint

        dst_endpoint = unicode('nersc#' + self.nersc_system)
        dst_path = self.spot.scratch_dir + '/' + file_name
        print src_endpoint, src_path, dst_endpoint, self.spot.scratch_dir
        r = self.globus.transfer_file(src_endpoint, src_path, dst_endpoint,
                                      dst_path)
        self.workingMessage.emit('Globus transfer sumbitted!')
        self.taskDone.emit()

    @QtCore.Slot(tuple)
    def handleSpotDownload(self, dset_info):
        dataset, stage, fpath, fname = map(str, dset_info)
        print ("[%s] Processing handleSpotDownload" %
               QtCore.QThread.currentThread().objectName())
        self.workingMessage.emit('Downloading dataset...')
        self.downloadSignal.emit(1)

        download = self.spot.download_dataset_generator(dataset, stage, fpath, fname)
        for frac in download:
            self.percentSignal.emit(frac*100)

        self.workingMessage.emit('Download finished.')
        self.taskDone.emit()