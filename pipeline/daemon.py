import os
import time
from PySide import QtCore
import multiprocessing
from xicam import debugtools
import numpy as np
from xicam import threads
import glob


def daemon(path,filter,newcallback,procold=False):


    @threads.iterator(newcallback)
    def checkdirectory():
        """
        Checks a directory for new files, comparing what files are there now vs. before
        """
        childfiles = set(os.listdir(path))

        while True:
            updatedchildren = set(glob.glob(os.path.join(path, filter)))
            newchildren = updatedchildren - childfiles
            childfiles = updatedchildren
            if newchildren: yield list(newchildren)
            time.sleep(.1)
    checkdirectory()

def chunks(l, n):
    """
    Yield successive n chunks from l.
    """
    chunksize = int(np.ceil(float(len(l)) / n))
    for i in xrange(n):
        yield l[i * chunksize:(i + 1) * chunksize]


def test(*args, **kwargs):
    print args, kwargs

if __name__ == '__main__':


    path = 'C:\\Users\\ronpa\\Downloads'
    # files = os.listdir(path)

    procold = False
    daemon(path, '*.tif', test, procold=procold)

    QtCore.QCoreApplication([]).exec_()