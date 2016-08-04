# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 17:22:00 2015

@author: lbluque
"""
import sys
import time
import functools
import Queue
import multiprocessing as mp
from PySide import QtCore
# Error is raised if this import is removed probably due to some circular with this module and something???
from client import spot, globus, sftp


QtCore.Signal = QtCore.Signal
QtCore.Slot = QtCore.Slot


class Emitter(QtCore.QObject):
    """
    Class that holds signals that can be emitted by a QRunnable
    """

    sigRetValue = QtCore.Signal(object)
    sigFinished = QtCore.Signal()
    sigExcept = QtCore.Signal(Exception, object)

    def __init__(self):
        super(Emitter, self).__init__()


class RunnableMethod(QtCore.QRunnable):
    """
    Runnable that will execute a given method from a QThreadPool and emit the response to the given callback function
    """

    def __init__(self, method, method_args=(), method_kwargs={}, callback_slot=None,
                 finished_slot=None, except_slot=None, priority=0, lock=None):
        """
        RunnableMethod constructor
        :param method: function object, method to run on seperate thread
        :param method_args: tup, positional arguments for method
        :param method_kwargs: dict, keyword arguments for method
        :param callback_slot: function object (qt slot), slot to recieve return value of method
        :param finished_slot: function object (qt slot), slot to recieve void signal if method finishes successfully
        :param except_slot: function object (qt slot), slot to recieve exception instance and traceback object
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
        if except_slot is not None:
            self.emitter.sigExcept.connect(except_slot, QtCore.Qt.QueuedConnection)

    def run(self):
        """
        Override virtual run method of QRunnable base class
        """

        # print 'Started {0} in thread {1}, will emit back to {2}'.format(self._method.__name__,
        #                                                                 QtCore.QThread.currentThread(),
        #                                                                 self._callback_slot.__name__)
        try:
            if self.lock is not None:
                self.lock.lock()
            value = self._method(*self.method_args, **self.method_kwargs)
            if value is None:
                value = False
            self.emitter.sigRetValue.emit(value)
        except Exception:
            ex_type, ex, tb = sys.exc_info()
            self.emitter.sigExcept.emit(ex, tb)
        else:
            self.emitter.sigFinished.emit()
        finally:
            if self.lock is not None:
                self.lock.unlock()


class RunnableIterator(RunnableMethod):
    """
    Runnable that will loop through an iterator and emit a signal representing the progress of a
    iterator or generator method
    """

    def __init__(self, iterator, iterator_args=(), iterator_kwargs={}, callback_slot=None,
                 finished_slot=None, interrupt_signal=None, except_slot=None, priority=0, lock=None):
        """
        RunnableIterator constructor
        :param iterator: generator object, method to run on seperate thread
        :param iterator_args: tup, positional arguments for generator
        :param iterator_kwargs: dict, keyword arguments for generator
        :param callback_slot: function object (qt slot), slot to recieve yield value of generator
        :param finished_slot: function object (qt slot), slot to recieve void signal if generator finishes successfully
        :param interrupt_signal:, qt signal, signal used to interrupt generator
        :param except_slot: function object (qt slot), slot to recieve exception instance and traceback object
        :param priority: int, priority given to runnable
        """
        super(RunnableIterator, self).__init__(method=iterator, method_args=iterator_args,
                                               method_kwargs=iterator_kwargs, callback_slot=callback_slot,
                                               finished_slot=finished_slot, except_slot=except_slot,
                                               priority=priority, lock=lock)
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
                    raise StopIteration('{0} running in background thread {1} interrupted'.format(
                                            self._method.__name__, QtCore.QThread.currentThreadId()))
                if status is None:
                    status = False
                self.emitter.sigRetValue.emit(status)
        except Exception:
            ex_type, ex, tb = sys.exc_info()
            self.emitter.sigExcept.emit(ex, tb)
        else:
            self.emitter.sigFinished.emit()
        finally:
            if self.lock is not None: self.lock.unlock()


def method(callback_slot=None, finished_slot=None, except_slot=None, lock=None):
    """
    Decorator for functions/methods to run as RunnableMethods on background QT threads
    Use it as any python decorator to decorate a function with @decorator syntax or at runtime:
    decorated_method = threads.method(callback_slot, ...)(method_to_decorate)
    then simply run it: decorated_iterator(*args, **kwargs)
    :param method: function/method to run on a background thread
    :param callback_slot: slot to call with the return value of the function
    :param finished_slot: slot to recieve finished signal when function completes
    :param except_slot: function object (qt slot), slot to recieve exception instance and traceback object
    :param lock: (mutex) simple lock if multiple access needs to be prevented
    :return: decorated method
    """

    def wrap_runnable_method(method):
        @functools.wraps(method)
        def _runnable_method(*args, **kwargs):
            runnable = RunnableMethod(method, method_args=args, method_kwargs=kwargs,
                                        callback_slot=callback_slot, finished_slot=finished_slot,
                                        except_slot=except_slot, lock=lock)
            add_to_queue(runnable)
        return _runnable_method
    return wrap_runnable_method


def iterator(callback_slot=None, finished_slot=None, interrupt_signal=None, except_slot=None, lock=None):
    """
    Decorator for iterators/generators to run as RunnableIterators on background QT threads
    Use it as any python decorator to decorate a function with @decorator syntax or at runtime:
    decorated_iterator = threads.iterator(callback_slot, ...)(iterator_to_decorate).
    then simply run it: decorated_iterator(*args, **kwargs)
    :param generator: iterator/generator to be decorated
    :param callback_slot: slot to call with the yield value or next value of iterator
    :param finished_slot: slot to receive finished signal when iterator finishes
    :param interrupt_signal: signal to break out of iterator loop prematurely
    :param except_slot: function object (qt slot), slot to recieve exception instance and traceback object
    :param lock: (mutex) simple lock if multiple access needs to be prevented
    :return: decorated iterator
    """
    def wrap_runnable_iterator(generator):
        @functools.wraps(generator)
        def _runnable_iterator(*args, **kwargs):
            runnable = RunnableIterator(generator, iterator_args=args, iterator_kwargs=kwargs,
                                        callback_slot=callback_slot, finished_slot=finished_slot,
                                        interrupt_signal=interrupt_signal, except_slot=except_slot, lock=lock)
            add_to_queue(runnable)
        return _runnable_iterator
    return wrap_runnable_iterator


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
queue = Queue.Queue()
worker = Worker(queue)
mutex = QtCore.QMutex()
worker_thread = QtCore.QThread()
worker.moveToThread(worker_thread)
worker_thread.started.connect(worker.run)


#  Only start worker if no debugger is being used
if not sys.gettrace():
    print 'Loading thread queue...'
    worker_thread.start()
