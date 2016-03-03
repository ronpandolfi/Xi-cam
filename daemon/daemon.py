import os
import time
import process
from PySide import QtCore
import multiprocessing
from xicam import debugtools


class daemon(QtCore.QThread):
    """
    This starts a daemon running as a QThread which watches a directory for new files and sends them to process.py
    Processing is done in separate Processes, splitting tasks between each core. Files that already exist can also
    be processed.
    """


    num_cores = multiprocessing.cpu_count()
    def __init__(self, path, experiment, procold=False):
        super(daemon, self).__init__()
        self.procold = procold
        self.experiment = experiment
        self.path = path
        self.exiting = False
        self.childfiles = set(os.listdir(path))


    def run(self):

        if self.procold:
            self.processfiles(self.path, self.childfiles)

        try:
            while not self.exiting:
                time.sleep(.1)
                self.checkdirectory()  # Force update; should not have to do this -.-
        except KeyboardInterrupt:
            pass

    def stop(self):
        self.exiting = True
        print ("thread stop - %s" % self.exiting)

    def __del__(self):
        self.exiting = True
        self.wait()

    @debugtools.timeit
    def processfiles(self, path, files):
        """
        distribute new files to cores for processing. Ignores .nxs.
        """

        files = [f for f in files if not os.path.splitext(f)[1] == '.hdf']
        if files:
            print os.path.splitext(path)[1]

            jobs = []
            p = None
            # filter paths
            files = list(chunks(files, self.num_cores))

            for i in range(self.num_cores):
                p = multiprocessing.Process(target=process.process, args=(path, files[i], self.experiment))
                jobs.append(p)
                p.start()

            while p.is_alive():  # TODO: change to wait for ALL threads?
                time.sleep(.1)



    def checkdirectory(self):
        """
        Checks a directory for new files, comparing what files are there now vs. before
        """
        updatedchildren = set(os.listdir(self.path))
        newchildren = updatedchildren - self.childfiles
        self.childfiles = updatedchildren
        self.processfiles(self.path, list(newchildren))



def chunks(l, n):
    """
    Yield successive n chunks from l.
    """
    chunksize = int(len(l) / n)
    for i in xrange(0, n, 1):
        yield l[i * chunksize:(i + 1) * chunksize]


if __name__ == '__main__':
    path = '/Users/rp/YL1031'
    # files = os.listdir(path)
    experiment = None
    procold = True
    d = daemon(path, experiment, procold)
    d.run()
