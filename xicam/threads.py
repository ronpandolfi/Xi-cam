# -*- coding: utf-8 -*-

__author__ = "Luis Barroso-Luque"
__copyright__ = "Copyright 2016, CAMERA, LBL, ALS"
__credits__ = ["Ronald J Pandolfi", "Dinesh Kumar", "Singanallur Venkatakrishnan", "Luis Luque", "Alexander Hexemer"]
__license__ = ""
__version__ = "1.2.1"
__maintainer__ = "Ronald J Pandolfi"
__email__ = "ronpandolfi@lbl.gov"
__status__ = "Beta"


import sys
import types
import traceback
import functools
import Queue
import multiprocessing as mp
from PySide import QtCore
from pipeline import msg
# Error is raised if this import is removed probably due to some circular with this module and something???
from client import spot, globus, sftp
from modpkgs import nonesigmod
from modpkgs import guiinvoker

class Emitter(QtCore.QObject):
    """
    Class that holds signals that can be emitted by a QRunnable

    Attributes
    ----------
    sigRetValue : QtCore.Signal
        Signal used to emit the return value from the function passed to Runnable
    sigFinised : QtCore.Signal
        Void signal emitted when the runnable has finished correctly (no exceptions raised)
    sigExcept : QtCore.Signal
        Signal that emits the exeption type, exception instance and traceback for an exception raised when running
        the Runnable
    """

    sigRetValue = QtCore.Signal(object)
    sigFinished = QtCore.Signal()
    sigExcept = QtCore.Signal(type, Exception, types.TracebackType)



def EmitterFactory(*sig):
    return type('Emitter',(Emitter,),{'sigTemp':QtCore.Signal(*sig)})

class RunnableMethod(QtCore.QRunnable):
    """
    Runnable that will execute a given method from a QThreadPool and emit the response to the given callback function
    To use simply instantiate and pass the instance to a QThreadPool.start()


    Attributes
    ----------
    emmiter : QtCore.QObject
        QObject used to emit signals to communicate with the main GUI thread
    _method : function/method
        Function/method that will be run in a background thread
    method_args : list
        List/tuple of arguments for the given function/method
    method_kwargs : dict
        Dictionary with keyword arguments for the give function/method
    lock : object
        Mutex used to lock when several threads are spawned
    _priority : int
        Priority given to this Runnable in the pool


    Parameters
    ----------
    method : function
        Function object, method to run on seperate thread
    method_args : list
        List/tuple of arguments for the given function/method
    method_kwargs : dict
        Dictionary with keyword arguments for the give function/method
    callback_slot : function
        Function/method to run on a background thread
    finished_slot : QtCore.Slot
        Slot to call with the return value of the function
    except_slot : QtCore.Slot
        Function object (qt slot), slot to receive exception type, instance and traceback object
    default_exhandle : bool
        Flag to use the default exception handle slot. If false it will not be called
    lock : mutex/semaphore
        Simple lock if multiple access needs to be prevented
    """

    def __init__(self, method, method_args=(), method_kwargs={}, callback_slot=None,
                 finished_slot=None, except_slot=None, default_exhandle=True, priority=0, lock=None):
        super(RunnableMethod, self).__init__()
        self.emitter = Emitter()
        self._method = method
        self.method_args = method_args
        self.method_kwargs = method_kwargs
        self._priority = priority
        self.lock = lock

        # Connect callback and finished slots to corresponding signals
        self._callback_slot = callback_slot
        if finished_slot is not None:
            self.emitter.sigFinished.connect(finished_slot, QtCore.Qt.QueuedConnection)
        if except_slot is not None:
            self.emitter.sigExcept.connect(except_slot, QtCore.Qt.QueuedConnection)
        if default_exhandle:
            self.default_handler = functools.partial(default_exception_handler)  # create a copy of the function
            self.emitter.sigExcept.connect(self.default_handler, QtCore.Qt.QueuedConnection)

    def run(self):
        """
        Override virtual run method of QRunnable base class to run the function/method that was passed in the
        constructor
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
            try:
                if self._callback_slot: self.emit(self._callback_slot, value)
            except RuntimeError:
                print 'this did not run'
                msg.logMessage(('Runnable method tried to return value, but signal was already disconnected.'),
                               msg.WARNING)
                if self.lock is not None: self.lock.unlock()
                return

        except Exception:
            etype, ex, tb = sys.exc_info()
            self.emitter.sigExcept.emit(etype, ex, tb)
        else:
            self.emitter.sigFinished.emit()
        finally:
            if self.lock is not None:
                self.lock.unlock()

    def emit(self,slot,*value):
        if str(type(slot)) == "<type 'PySide.QtCore.SignalInstance'>": # allows slotting into signals; this type is not in QtCore, so must compare by name str
            if slot is None: return
            value = map(nonesigmod.pyside_none_wrap, value)
            tempemitter = EmitterFactory(*[object] * len(value))()
            tempemitter.sigTemp.connect(slot, QtCore.Qt.QueuedConnection)
            tempemitter.sigTemp.emit(*value)
        else:
            guiinvoker.invoke_in_main_thread(slot, *value) # actually works better than communicating with signals





class RunnableIterator(RunnableMethod):
    """
    Runnable that will loop through an iterator and emit a signal representing the progress of a
    iterator or generator method.

    See RunnableMethod for attributes


    Attributes
    ----------
    interrupt_signal : QtCore.Signal
        Signal used to interrupt generator


    Parameters
    ----------
    iterator : iterator
        Generator/iterator object to loop through on seperate thread
    iterator_args : list
        List/tuple of arguments for the given function/method
    iterator_kwargs : dict
        Dictionary with keyword arguments for the give function/method
    callback_slot : function
        Function/method to run on a background thread
    finished_slot : QtCore.Slot
        Slot to call with the return value of the function
    interrupt_signal : QtCore.Signal
        Signal used to interrupt generator
    except_slot : QtCore.Slot
        Function object (qt slot), slot to receive exception type, instance and traceback object
    default_exhandle : bool
        Flag to use the default exception handle slot. If false it will not be called
    lock : mutex/semaphore
        Simple lock if multiple access needs to be prevented
    parent : object
        Parent object reference; keeps the parent from being garbage collected before a callback. If the parent is ready
        for GC'ing, the iterator will stop, and should free the final reference.
    """

    def __init__(self, iterator, iterator_args=(), iterator_kwargs={}, callback_slot=None,
                 finished_slot=None, interrupt_signal=None, except_slot=None, default_exhandle=True,
                 priority=0, lock=None, parent=None):
        super(RunnableIterator, self).__init__(method=iterator, method_args=iterator_args,
                                               method_kwargs=iterator_kwargs, callback_slot=callback_slot,
                                               finished_slot=finished_slot, except_slot=except_slot,
                                               default_exhandle=default_exhandle, priority=priority, lock=lock)
        self._interrupt = False
        self._callback_slot = callback_slot
        self._parent = parent

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
                if type(status) is not tuple: status = (status,)
                if self._interrupt:
                    raise StopIteration('{0} running in background thread {1} interrupted'.format(
                                            self._method.__name__, QtCore.QThread.currentThreadId()))
                try:
                    # print 'status:',status
                    self.emit(self._callback_slot,*status)
                    # self.emitter.sigRetValue.emit(status)
                except RuntimeError:
                    msg.logMessage(('Runnable iterator tried to return value, but signal was already disconnected.'),msg.WARNING)
                    if self.lock is not None: self.lock.unlock()
                    return
        except Exception:
            etype, ex, tb = sys.exc_info()
            self.emitter.sigExcept.emit(etype, ex, tb)
        else:
            self.emitter.sigFinished.emit()
        finally:
            if self.lock is not None: self.lock.unlock()


def method(callback_slot=None, finished_slot=None, except_slot=None, default_exhandle=True, lock=None):
    """
    Decorator for functions/methods to run as RunnableMethods on background QT threads
    Use it as any python decorator to decorate a function with @decorator syntax or at runtime:
    decorated_method = threads.method(callback_slot, ...)(method_to_decorate)
    then simply run it: decorated_iterator(*args, **kwargs)

    Parameters
    ----------
    callback_slot : function
        Function/method to run on a background thread
    finished_slot : QtCore.Slot
        Slot to call with the return value of the function
    except_slot : QtCore.Slot
        Function object (qt slot), slot to receive exception type, instance and traceback object
    default_exhandle : bool
        Flag to use the default exception handle slot. If false it will not be called
    lock : mutex/semaphore
        Simple lock if multiple access needs to be prevented

    Returns
    -------
    wrap_runnable_method : function
        Decorated function/method
    """

    def wrap_runnable_method(function):
        @functools.wraps(function)
        def _runnable_method(*args, **kwargs):
            runnable = RunnableMethod(function, method_args=args, method_kwargs=kwargs,
                                      callback_slot=callback_slot, finished_slot=finished_slot,
                                      except_slot=except_slot, default_exhandle=default_exception_handler, lock=lock)
            add_to_queue(runnable)
        return _runnable_method
    return wrap_runnable_method


def iterator(callback_slot=None, finished_slot=None, interrupt_signal=None, except_slot=None, lock=None, parent=None):
    """
    Decorator for iterators/generators to run as RunnableIterators on background QT threads
    Use it as any python decorator to decorate a function with @decorator syntax or at runtime:
    decorated_iterator = threads.iterator(callback_slot, ...)(iterator_to_decorate).
    then simply run it: decorated_iterator(*args, **kwargs)

    Parameters
    ----------
    callback_slot : function
        Function/method to run on a background thread
    finished_slot : QtCore.Slot
        Slot to call with the return value of the function
    interrupt_signal : QtCore.Signal
        Signal to break out of iterator loop prematurely
    except_slot : QtCore.Slot
        Function object (qt slot), slot to receive exception type, instance and traceback object
    lock : mutex/semaphore
        Simple lock if multiple access needs to be prevented

    Returns
    -------
    wrap_runnable_iterator : function
        Decorated iterator/generator
    """

    def wrap_runnable_iterator(generator):
        @functools.wraps(generator)
        def _runnable_iterator(*args, **kwargs):
            runnable = RunnableIterator(generator, iterator_args=args, iterator_kwargs=kwargs,
                                        callback_slot=callback_slot, finished_slot=finished_slot,
                                        interrupt_signal=interrupt_signal, except_slot=except_slot,
                                        lock=lock, parent=parent)
            add_to_queue(runnable)
        return _runnable_iterator
    return wrap_runnable_iterator


class Worker(QtCore.QThread):
    """
    Daemon worker that contains a Queue and QThreadPool for running jobs

    Attributes
    ----------
    queue : Queue
        Queue to put/get runnables
    pool : QThreadPool
        Thread pool used to run runnables from queue

    Parameters
    ----------
    queue : Queue
        Queue to put/get runnables
    """

    def __init__(self, queue, parent=None):
        super(Worker, self).__init__(parent)
        self.queue = queue
        self.pool = QtCore.QThreadPool.globalInstance()


    # def __del__(self):
    #     self.queue.join()

    def stop(self):
        """
        Puts a None in the queue to signal the run loop to exit
        """

        self.queue.put(None)
        self.quit()
        print 'threads:',self.pool.activeThreadCount()

    def run(self):
        """
        Continuously get Runnables from queue and running them on available threads as long as stop flag is false
        """
        while True:
            runnable = self.queue.get()
            # print "Worker got item {} off queue".format(type(runnable))
            if runnable is None:
                break
            self.pool.start(runnable, runnable._priority)
            self.queue.task_done()


@QtCore.Slot(type, Exception, types.TracebackType)
def default_exception_handler(etype, ex, tb):
    """
    Default exception handle slot that will log the exception and traceback to the logger from pipeline.msg
    this will also print the exception

    Parameters
    ----------
    etype : Exception type
        Type of the exception being raised from background thread
    ex : Exception
        Exception object raised from the function sent to a background thread
    tb : Traceback
        Traceback object raised from the function sent to a background thread

    Returns
    -------
    None
    """

    exmessage = ''.join(traceback.format_exception(etype, ex, tb))  # concatenate the list of strings
    msg.logMessage(exmessage, level=msg.ERROR)
    traceback.print_exception(etype, ex, tb)


def add_to_queue(runnable):
    """
    Add a Runnable to the global thread.queue

    Parameters
    ----------
    runnable : QtCore.Runnable

    """
    global queue
    try:
        queue.put(runnable)
    except Exception as e:
        msg.logMessage(e, level=msg.ERROR)
        raise e


# Application globals
queue = Queue.Queue()
worker = Worker(queue)
mutex = QtCore.QMutex()
worker.start()
