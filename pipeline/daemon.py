import os
import time
from PySide import QtCore
import multiprocessing
import numpy as np
import glob

from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler


class Watcher(Observer):
    def __init__(self, path, filter, newcallback, procold=False, recursive=True):
        self.filter = filter
        self.newcallback = newcallback
        self.procold = procold

        event_handler = PatternMatchingEventHandler(patterns=[filter])
        event_handler.on_created = newcallback
        super(Watcher, self).__init__()
        self.schedule(event_handler, path, recursive=recursive)
        self.start()


def chunks(l, n):
    """
    Yield successive n chunks from l.
    """
    chunksize = int(np.ceil(float(len(l)) / n))
    for i in xrange(n):
        yield l[i * chunksize:(i + 1) * chunksize]


def test(*args, **kwargs):
    print(args, kwargs)


if __name__ == '__main__':
    path = '/home/rp/data/test'
    # files = os.listdir(path)

    procold = True
    d = Watcher(path, '*.tif', test, procold=procold)
    d.start()
    print('here?')

    QtCore.QCoreApplication([]).exec_()
