# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 17:22:00 2015

@author: lbluque
"""

import time
import Queue
import multiprocessing as mp
from PySide import QtCore
# Error is raised if this import is removed probably due to some circular import between xglobals and here
from client import spot, globus


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
        self.lock = None
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
            if self.lock is not None: self.lock.lock()
            value = self._method(*self.args, **self.kwargs)
        except Exception:
            raise
        finally:
            if self.lock is not None: self.lock.unlock()

        if value is None:
            value = False

        self.emitter.sigRetValue.emit(value)

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
        self.pool = QtCore.QThreadPool.globalInstance()  # Should I use globalInstance() or a seperate instance?
        self.pool.setMaxThreadCount(mp.cpu_count())
        self._stop = False

    def __del__(self):
        self.queue.join()

    def startRunnable(self, runnable, priority=0):
        self.pool.start(runnable, priority)

    def stopWork(self):
        self._stop = True

    def run(self):
        while not self._stop:
            item = self.queue.get()
            # print "Worker got item {} off queue".format(type(item))
            self.startRunnable(item)
            self.queue.task_done()
            time.sleep(0.1)

#TODO: allow threads to be compatibile with debugging
# Application globals
import sys
if not sys.gettrace():
    print 'Loading thread queue...'
    global queue, worker, worker_thread, mutex
    queue = Queue.Queue()
    worker = Worker(queue)
    mutex = QtCore.QMutex()
    worker_thread = QtCore.QThread()
    worker.moveToThread(worker_thread)
    worker_thread.started.connect(worker.run)
    worker_thread.start()
