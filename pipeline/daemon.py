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
        childfiles = set(glob.glob(os.path.join(path, filter)))
        if procold and childfiles: yield list(childfiles)

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


    path = '/home/rp/data/HiTp/testset/'
    # files = os.listdir(path)

    procold = True
    daemon(path, '*.tif', test, procold=procold)

    QtCore.QCoreApplication([]).exec_()