import sys
import os
import time
import logging
import process
from PySide import QtCore, QtGui
from hipies import watcher

from watchdog import events
from watchdog import observers
from joblib import Parallel, delayed
import multiprocessing
from hipies import debug

# http://stackoverflow.com/questions/6783194/background-thread-with-qthread-in-pyqt


class daemon(QtCore.QThread):
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

            while True:
                time.sleep(.1)
                self.checkdirectory()  # Force update; should not have to do this -.-
        except KeyboardInterrupt:
            pass

    @debug.timeit
    def processfiles(self, path, files):
        if files:
            print os.path.splitext(path)[1]


            # process.process([os.path.join(path, f)], self.experiment)
            jobs = []
            p = None
            # filter paths
            files = [f for f in files if not os.path.splitext(f)[1] == '.nxs']
            files = list(chunks(files, self.num_cores))

            for i in range(self.num_cores):
                p = multiprocessing.Process(target=process.process, args=(path, files[i], self.experiment))
                jobs.append(p)
                p.start()

            while p.is_alive():
                time.sleep(.1)


                # print('here:',os.path.join(path,file))

    def checkdirectory(self):
        # print(path)
        updatedchildren = set(os.listdir(self.path))
        newchildren = updatedchildren - self.childfiles
        self.childfiles = updatedchildren
        self.processfiles(self.path, list(newchildren))



# class newfilehandler(events.PatternMatchingEventHandler):
# patterns = ['*.edf']
#     def __init__(self,experiment):
#         super(newfilehandler, self).__init__()
#         self.experiment=experiment
#
#     def doprocessing(self,event):
#         #if event.event_type in ['modified','created']:
#         print event.src_path
#         process.process([event.src_path],self.experiment)
#
#     def on_created(self,event):
#         self.doprocessing(event)
#
#     def on_modified(self,event):
#         self.doprocessing(event)


def chunks(l, n):
    """ Yield successive n chunks from l.
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
