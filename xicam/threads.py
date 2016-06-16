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

    def __init__(self, method, method_args=(), method_kwargs={}, callback_slot=None, finished_slot=None,
                 priority=0, lock=None):
        """
        RunnableMethod constructor
        :param method: function object, method to run on seperate thread
        :param method_args: tup, positional arguments for method
        :param method_kwargs: dict, keyword arguments for method
        :param callback_slot: function object (qt slot), slot to recieve return value of method
        :param finished_slot: function object (qt slot), slot to recieve void signal if method finishes successfully
        :param priority: int, priority given to runnable
        """
        super(RunnableMethod, self).__init__()
        self.emitter = Emitter()
        self._method = method
        self._priority = priority
        self.lock = lock
        self.method_args = method_args
        self.method_kwargs = method_kwargs

        self.setAutoDelete(True)

        # Connect callback and finished slots to corresponding signals
        if callback_slot is not None:
            self.emitter.sigRetValue.connect(callback_slot, QtCore.Qt.QueuedConnection)
        if finished_slot is not None:
            self.emitter.sigFinished.connect(finished_slot, QtCore.Qt.QueuedConnection)

    def run(self):
        """
        Override virtual run method of QRunnable base class
        """

        # print 'Started {0} in thread {1}, will emit back to {2}'.format(self._method.__name__,
        #                                                                 QtCore.QThread.currentThread(),
        #                                                                 self._callback_slot.__name__)
        try:
            if self.lock is not None: self.lock.lock()
            value = self._method(*self.method_args, **self.method_kwargs)
            if value is None:
                value = False
            self.emitter.sigRetValue.emit(value)
        except Exception as ex:
            raise ex
            print 'Error: ', ex.message
        else:
            self.emitter.sigFinished.emit()
        finally:
            if self.lock is not None: self.lock.unlock()


class RunnableIterator(RunnableMethod):
    """
    Runnable that will loop through an iterator and emit a signal representing the progress of a generator method
    """

    def __init__(self, generator, generator_args=(), generator_kwargs={},
                 callback_slot=None, finished_slot=None, interrupt_signal=None, priority=0, lock=None):
        """
        RunnableIterator constructor
        :param generator: generator object, method to run on seperate thread
        :param generator_args: tup, positional arguments for generator
        :param generator_kwargs: dict, keyword arguments for generator
        :param callback_slot: function object (qt slot), slot to recieve yield value of generator
        :param finished_slot: function object (qt slot), slot to recieve void signal if generator finishes successfully
        :param interrupt_signal:, qt signal, signal used to interrupt generator
        :param priority: int, priority given to runnable
        """
        super(RunnableIterator, self).__init__(generator, generator_args,
                                               generator_kwargs, callback_slot, finished_slot, priority, lock)
        self._interrupt = False

        if interrupt_signal is not None:
            interrupt_signal.connect(self.interrupt)

    def interrupt(self):
        self._interrupt = True

    def run(self):
        """
        Override virtual run method of QRunnable base class
        """

        # print 'Started {0} in thread {1}, will update to {2}'.format(self._method.__name__,
        #                                                              QtCore.QThread.currentThread(),
        #                                                              self._callback_slot.__name__)
        if self.lock is not None: self.lock.lock()
        try:
            for status in self._method(*self.method_args, **self.method_kwargs):
                if self._interrupt:
                    raise StopIteration('{0} running in background thread {1} interrupted'.format(self._method.__name__,
                                                                                    QtCore.QThread.currentThreadId()))
                if status is None:
                    status = False
                self.emitter.sigRetValue.emit(status)
        except Exception as ex:
            print 'Error: ', ex.message
        else:
            self.emitter.sigFinished.emit()
        finally:
            if self.lock is not None: self.lock.unlock()



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

    def stopWork(self):
        self._stop = True

    def run(self):
        """
        Continuously get Runnables from queue and running them on available threads as long as stop flag is false
        """
        while not self._stop:
            runnable = self.queue.get()
            # print "Worker got item {} off queue".format(type(item))
            self.pool.start(runnable, runnable._priority)
            self.queue.task_done()
            time.sleep(0.1)


def add_to_queue(runnable):
    global queue
    try:
        queue.put(runnable)
    except Exception as e:
        print 'Error: ', e.message


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
