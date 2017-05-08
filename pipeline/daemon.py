import os
import time
from PySide import QtCore
import multiprocessing
from xicam import debugtools
import numpy as np
from xicam import threads
import glob

class daemon(QtCore.QRunnable):
    """
    This starts a daemon running as a QThread which watches a directory for new files and sends them to process.py
    Processing is done in separate Processes, splitting tasks between each core. Files that already exist can also
    be processed.
    """


    num_cores = multiprocessing.cpu_count()
    def __init__(self, path, filter, newcallback, procold=False):
        super(daemon, self).__init__()
        self._priority = 0
        self.procold = procold
        self.path = path
        self.exiting = False
        self.childfiles = set(os.listdir(path))
        self.newcallback = newcallback
        self.filter = filter
        threads.add_to_queue(self)



    def run(self):

        if self.procold:
            self.processfiles(self.path, self.childfiles)

        try:
            while not self.exiting:
                time.sleep(1)
                self.checkdirectory()  # Force update; should not have to do this -.-
        except KeyboardInterrupt:
            pass

    def stop(self):
        self.exiting = True
        print ("thread stop - %s" % self.exiting)

    def __del__(self):
        self.exiting = True
        self.wait()

    def processfiles(self, path, files):
        """
        distribute new files to cores for processing. Ignores .nxs.
        """
        print files

        files = [f for f in files] # match here
        if files:

            jobs = []
            p = None
            # filter paths
            files = list(chunks(files, self.num_cores))
            print files

            for i in range(self.num_cores):
                p = multiprocessing.Process(target=self.newcallback, args=(files[i],))
                jobs.append(p)
                p.start()

            while p.is_alive():  # TODO: change to wait for ALL threads?
                time.sleep(.1)



    def checkdirectory(self):
        """
        Checks a directory for new files, comparing what files are there now vs. before
        """
        updatedchildren = set(glob.glob(os.path.join(self.path,self.filter)))
        newchildren = updatedchildren - self.childfiles
        self.childfiles = updatedchildren
        if newchildren: self.processfiles(self.path, list(newchildren))



def chunks(l, n):
    """
    Yield successive n chunks from l.
    """
    chunksize = int(np.ceil(float(len(l)) / n))
    for i in xrange(n):
        yield l[i * chunksize:(i + 1) * chunksize]


if __name__ == '__main__':
    path = '/Users/rp/YL1031'
    # files = os.listdir(path)
    experiment = None
    procold = True
    d = daemon(path, experiment, procold)
    d.run()
