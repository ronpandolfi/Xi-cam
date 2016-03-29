# -*- coding: utf-8 -*-
import os
import sys

from PySide import QtGui,QtCore

from spew.spewwindow import SpewApp
from spew import threads

def main():
    sys.path.append(os.getcwd())
    app = QtGui.QApplication(sys.argv)

    window = SpewApp(app)
    window.show()
    window.raise_()
    window.activateWindow()
    app.setActiveWindow(window)
    threads.worker = threads.Worker(threads.queue)
    threads.worker_thread = QtCore.QThread(objectName='workerThread')
    threads.worker.moveToThread(threads.worker_thread)
    threads.worker_thread.started.connect(threads.worker.run)
    #worker_thread.start()
    QtCore.QThread.currentThread().setObjectName('main')
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()