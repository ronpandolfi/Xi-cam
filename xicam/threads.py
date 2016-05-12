# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 17:22:00 2015

@author: lbluque
"""

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
    sigFinished = QtCore.Signal()

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

        if callback_slot is not None:
            self.emitter.sigRetValue.connect(self._callback_slot, QtCore.Qt.QueuedConnection)

    def run(self):
        # print 'Started {0} in thread {1}, will emit back to {2}'.format(self._method.__name__,
        #                                                                 QtCore.QThread.currentThread(),
        #                                                                 self._callback_slot.__name__)
        # self.emitter.sigFinished.connect(self._callback_slot)  # Connect here or in constructor?
        try:
            value = self._method(*self.args, **self.kwargs)
            if value is None:
                value = False
            self.emitter.sigRetValue.emit(value)
        except Exception, e:
            raise e
        finally:
            self.emitter.sigFinished.emit()


class RunnableIterator(RunnableMethod):
    """
    Runnable that will loop through an iterator and emit a signal representing the progress of a generator method
    """

    def __init__(self, callback_slot, generator, *args, **kwargs):
        super(RunnableIterator, self).__init__(callback_slot, generator, *args, **kwargs)

    def run(self):
        # print 'Started {0} in thread {1}, will update to {2}'.format(self._method.__name__,
        #                                                              QtCore.QThread.currentThread(),
        #                                                              self._callback_slot.__name__)
        for status in self._method(*self.args, **self.kwargs):
            self.emitter.sigRetValue.emit(status)

        self.emitter.sigFinished.emit()


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
global queue, worker
queue = Queue.Queue()
worker = Worker(queue)
worker_thread = QtCore.QThread(objectName='workerThread')
worker.moveToThread(worker_thread)
worker_thread.started.connect(worker.run)
