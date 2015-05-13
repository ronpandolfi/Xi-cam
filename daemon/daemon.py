import sys
import os
import time
import logging
import process
from PySide import QtCore, QtGui
import watcher

from watchdog import events
from watchdog import observers


class daemon(QtCore.QThread):
    def __init__(self, path, experiment):
        super(daemon, self).__init__()
        self.experiment = experiment
        self.path = path
        self.exiting = False

    def run(self):
        # event_handler=newfilehandler(self.experiment)
        # observer = observers.Observer()
        # observer.schedule(event_handler, self.path, recursive=True)
        # observer.start()
        self.watcher = watcher.newfilewatcher()
        self.watcher.addPath(self.path)
        self.watcher.newFilesDetected.connect(self.processfiles)
        try:

            while True:
                self.msleep(1000)
                self.watcher.checkdirectory(self.path)  # Force update; should not have to do this -.-
        except KeyboardInterrupt:
            self.watcher = None
            # observer.stop()
            # observer.join()

    def processfiles(self, path, files):
        for file in files:
            print os.path.splitext(path)[1]
            if not os.path.splitext(file)[1] == '.nxs':
                print('Processing new file: ' + file)
                process.process([os.path.join(path, file)], self.experiment)
                # print('here:',os.path.join(path,file))


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
