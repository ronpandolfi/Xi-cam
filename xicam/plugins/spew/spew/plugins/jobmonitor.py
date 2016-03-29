# -*- coding: utf-8 -*-
"""
@author: lbluque
"""

from PySide import QtGui, QtCore
import base
from spew import threads, plugins
from spew.plugins.widgets import reconwizard

QtCore.Signal = QtCore.Signal
QtCore.Slot = QtCore.Slot


class Plugin(base.Plugin):
    """
    Class for job monitoring plugin
    """

    name = 'Job Monitor'

    def __init__(self, placeholders, parent=None):
        self.parent = parent
        self.jobs = []

        self.centerwidget = QtGui.QTableWidget(0, 3)
        self.centerwidget.setStyleSheet('background-color:#212121;')
        self.centerwidget.setHorizontalHeaderLabels(['Type', 'Progress', 'Description'])
        self.centerwidget.setFrameShape(QtGui.QFrame.NoFrame)
        self.centerwidget.verticalHeader().hide()
        self.centerwidget.horizontalHeader().setClickable(False)
        self.centerwidget.horizontalHeader().setStretchLastSection(True)
        self.centerwidget.horizontalHeader().setDefaultAlignment(QtCore.Qt.AlignLeft)
        self.centerwidget.setColumnWidth(0, 100)
        self.centerwidget.setColumnWidth(1, 400)
        self.centerwidget.setColumnWidth(2, 60)
        #self.centerwidget.horizontalHeader().setResizeMode(QtGui.QHeaderView.Stretch)
        self.centerwidget.setShowGrid(False)

        # Think about these conections....
        base.fileexplorer.sigExternalJob.connect(self.addRunnableJob)
        #reconwizard.Wizard.sigExternalJob.connect(self.addPopenJob)

        super(Plugin, self).__init__(placeholders, parent)

    def addJob(self, job_type, job_desc):
        row_num = self.centerwidget.rowCount()
        self.centerwidget.insertRow(row_num)
        jobentry = JobEntry(job_type)
        jobentry.setDescription(job_desc)
        self.jobs.append(jobentry)
        self.centerwidget.setItem(row_num, 0, jobentry.job_type)
        self.centerwidget.setCellWidget(row_num, 1, jobentry.widget)
        self.centerwidget.setCellWidget(row_num, 2, jobentry.desc_label)
        return jobentry

    @QtCore.Slot(str, str, object, list, dict)
    def addRunnableJob(self, job_type, job_desc, method, args, kwargs):
        job_entry = self.addJob(job_type, job_desc)
        runnable = threads.RunnableIterator(job_entry.progress, method, *args, **kwargs)
        threads.queue.put(runnable)
        #jobentry.sigCancel.connect(lambda x: self.removeJob(x))
        #return jobentry

    @QtCore.Slot(str, str, object, list, dict)
    def addPopenJob(self,  job_type, job_desc, method, args, kwargs):
        job_entry = self.addJob(job_type, job_desc)
        proc = method(*args, **kwargs)
        job_entry.pulseStart()
        return proc

    def removeJob(self, jobentry):
        idx = self.jobs.index(jobentry)
        del self.jobs[idx]
        self.centerwidget.removeRow(idx)
        jobentry.deleteLater()


class JobEntry(QtGui.QWidget):
    """
    Job entries
    """

    sigCancel = QtCore.Signal(object)

    def __init__(self, job_type):
        super(JobEntry, self).__init__()
        self.job_type = QtGui.QTableWidgetItem(job_type)
        self.job_type.setFlags(QtCore.Qt.ItemIsEnabled)
        # self.job_type.setTextAlignment(QtCore.Qt.AlignCenter)
        self.desc_label = QtGui.QLabel()
        self.progressbar = QtGui.QProgressBar()
        self.progressbar.setRange(0, 100)
        self.cancel_button = QtGui.QToolButton()
        self.cancel_button.setFixedSize(32, 32)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap('gui/cancel.png'), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.cancel_button.setIcon(icon)
        self.cancel_button.setIconSize(QtCore.QSize(32, 32))
        self.widget = QtGui.QWidget()
        self.widget.setLayout(QtGui.QHBoxLayout())
        self.widget.layout().setContentsMargins(0, 0, 0, 0)
        self.widget.layout().addWidget(self.progressbar)
        self.widget.layout().addWidget(self.cancel_button)

        self.cancel_button.pressed.connect(self.cancelPressed)

    def setDescription(self, desc):
        self.desc_label.setText(desc)

    def cancelPressed(self):
        self.sigCancel.emit(self)

    def progress(self, i):
        i = int(i*100)
        self.progressbar.setValue(i)

    def pulseStart(self):
        self.progressbar.setRange(0, 0)

    def pulseStop(self):
        self.progressbar.setRange(0, 1)
        self.progress(1)

