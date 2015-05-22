import sys
import os
import time
import logging
import process
from PySide import QtCore, QtGui
from hipies import watcher

from watchdog import events
from watchdog import observers

# http://stackoverflow.com/questions/6783194/background-thread-with-qthread-in-pyqt


class daemon(QtCore.QThread):
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
                time.sleep(1)
                self.checkdirectory()  # Force update; should not have to do this -.-
        except KeyboardInterrupt:
            pass

    def processfiles(self, path, files):
        for f in files:
            print os.path.splitext(path)[1]
            if not os.path.splitext(f)[1] == '.nxs':
                print('Processing new file: ' + f)
                process.process([os.path.join(path, f)], self.experiment)
                # print('here:',os.path.join(path,file))

    def checkdirectory(self):
        # print(path)
        updatedchildren = set(os.listdir(self.path))
        newchildren = updatedchildren - self.childrenfiles
        self.childrenfiles = updatedchildren
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
