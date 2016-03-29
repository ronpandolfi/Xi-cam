# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 17:22:00 2015

@author: lbluque
"""

import os
from PySide import QtGui, QtCore
from os.path import expanduser
from collections import OrderedDict
from spew import threads
from spew.plugins.widgets import login
from pipeline import loader

QtCore.Signal = QtCore.Signal
QtCore.Slot = QtCore.Slot


class LocalFileView(QtGui.QTreeView):
    """
    Local file explorer tree view
    """

    pathChanged = QtCore.Signal(str)
    sigOpenFiles = QtCore.Signal(list)
    sigOpenDir = QtCore.Signal(str)

    def __init__(self, parent=None):
        super(LocalFileView, self).__init__(parent)

        self.file_model = QtGui.QFileSystemModel()
        self.mode = self.file_model
        self.setModel(self.file_model)
        self.path = expanduser('~')
        self.refresh(self.path)

        header = self.header()
        self.setHeaderHidden(True)
        for i in range(1, 4):
            header.hideSection(i)
        filefilter = map(lambda s: '*'+s,loader.acceptableexts)
        self.file_model.setNameFilters(filefilter)
        self.file_model.setNameFilterDisables(False)
        self.file_model.setResolveSymlinks(True)
        self.expandAll()
        self.sortByColumn(0, QtCore.Qt.AscendingOrder)
        self.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        self.setSelectionMode(self.ExtendedSelection)
        self.setIconSize(QtCore.QSize(16, 16))
#        self.setContextMenuPolicy(Qt.CustomContextMenu)
#        self.customContextMenuRequested.connect(self.contextMenu)

        self.doubleClicked.connect(self.onDoubleClick)

    def refresh(self, path=None):
        """
        Refresh the file tree, or switch directories and refresh
        """
        if path is None:
            path = self.filetreepath

        root = QtCore.QDir(path)
        self.file_model.setRootPath(root.absolutePath())
        self.setRootIndex(self.file_model.index(root.absolutePath()))
        self.path = path
        self.pathChanged.emit(path)

    def onDoubleClick(self, index):
        item = self.file_model.index(index.row(), 0, index.parent())
        path = self.file_model.filePath(item)

        if os.path.isdir(path):
            self.refresh(path=path)
        else:
            self.sigOpenFiles.emit([path])

    def getSelectedFilePath(self):
        selected =  str(self.file_model.filePath(self.currentIndex()))
        if selected == '':
            selected = None
        return selected

    def getSelectedFile(self):
        selected = str(self.file_model.fileName(self.currentIndex()))
        if selected == '':
            selected = None
        return selected

    def downloadFile(self):
        return None

    def transferFile(self):
        return None

    def uploadFile(self):
        fpath = self.getSelectedFilePath()
        return None

    def deleteFile(self):
        self.file_model.remove(self.currentIndex())


class FileExplorer(QtGui.QWidget):
    """
    Container for local file tree view that adds back button and path label
    """

    def __init__(self, file_view, parent=None):
        super(FileExplorer, self).__init__(parent)

        self.file_view = file_view
        self.path_label = QtGui.QLineEdit(self)
        self.back_button = QtGui.QToolButton(self)

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap('gui/spew/back.png'), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.back_button.setIcon(icon)
        self.back_button.setIconSize(QtCore.QSize(18, 18))
        self.back_button.setFixedSize(32, 32)

        self.path_label.setReadOnly(True)

        l = QtGui.QVBoxLayout(self)
        l.setStretch(0, 0)
        l.setContentsMargins(0, 0, 0, 0)
        l.setSpacing(0)
        hl = QtGui.QHBoxLayout()
        hl.addWidget(self.back_button)
        hl.addWidget(self.path_label)
        hl.setStretch(0, 0)
        l.addLayout(hl)
        l.addWidget(self.file_view)

        self.setPathLabel(self.file_view.path)
        self.back_button.clicked.connect(self.onBackClick)
        self.file_view.pathChanged.connect(self.setPathLabel)

    def onBackClick(self):
        path = self.file_view.path
        path = os.path.dirname(str(path))
        self.file_view.refresh(path)

    def setPathLabel(self, path):
        self.path_label.setText(path)

    def getRawDatasetList(self):
        widget = self.file_view
        items = widget.findItems('*.h5', QtCore.Qt.MatchWildcard)
        items = [i.text() for i in items]
        return items

    def getSelectedFilePath(self):
        return self.file_view.getSelectedFilePath()


class RemoteFileView(QtGui.QListWidget):
    """
    Remote file explorer (NERSC and Globus)
    """
    pathChanged = QtCore.Signal(str)

    def __init__(self, remote_client, parent=None):
        super(RemoteFileView, self).__init__(parent)
        self.path = None
        self.client = remote_client
        self.itemDoubleClicked.connect(self.onDoubleClick)

    def onDoubleClick(self, item):
        file_name = item.text()
        if len(file_name.split('.')) == 1:
            path = self.path + '/' + str(file_name)
            self.refresh(path)

        elif '.h5' in file_name:
            # get file metadata here dudley boy
            print file_name + " is an h5 file!"

    def getSelectedFilePath(self):
        return '{0}/{1}'.format(self.path, str(self.currentItem().text()))

    def getSelectedFile(self):
        return str(self.currentItem().text())

    def getDirContents(self, path, *args):
        runnable = threads.RunnableMethod(self.fillList, self.client.get_dir_contents, path, *args)
        threads.queue.put(runnable)

    def refresh(self, path):
        self.path = path
        self.getDirContents(self.path)
        self.pathChanged.emit(path)

    def fillList(self, value):
        self.clear()
        for item in value:
            if item['name'] != '.' and item['name'] != '..':
                self.addItem(item['name'])

    def downloadFile(self):
        fpath = self.getSelectedFilePath()
        fname = self.getSelectedFile()
        fileDialog = QtGui.QFileDialog(self, 'Save as', os.path.expanduser('~'))
        fileDialog.setAcceptMode(QtGui.QFileDialog.AcceptSave)
        fileDialog.selectFile(fname)
        if fileDialog.exec_():
            outpath = str(fileDialog.selectedFiles()[0])
            return fpath, outpath

    def transferFile(self):
        return None

    def uploadFile(self):
        return None

    def deleteFile(self):
        return None


class NERSCFileView(RemoteFileView):
    """
    File explorer for NERSC systems, must be passed a client and a worker to make REST calls
    """

    def __init__(self, nersc_client, system, parent=None):
        self.system = system
        super(NERSCFileView, self).__init__(nersc_client, parent=parent)
        self.client.set_scratch_dir(system)
        self.path = self.client.scratch_dir
        self.getDirContents(self.path, self.system)

    def refresh(self, path):
        self.path = path
        super(NERSCFileView, self).getDirContents(path, self.system)
        self.pathChanged.emit(path)

    def downloadFile(self):
        fpath, outpath = super(NERSCFileView, self).downloadFile()
        size = self.client.get_file_size(fpath, self.system)
        desc = 'File {0} from {1}'.format(fpath, self.system)
        if size < 100*2**20:
            method = self.client.download_file_generator
            outpath, fname = os.path.split(outpath)
            args = [fpath, self.system, outpath]
            kwargs = {'fname': fname}
        else:
            # USE GLOBUS API CALL HERE, Think of how to get access to client
            pass

        return desc, method, args, kwargs

    def transferFile(self):
        return None

    def uploadFile(self):
        return None

    def deleteFile(self):
        return None


class GlobusFileView(RemoteFileView):
    """
    File explorer for Globus endpoints
    """

    def __init__(self, endpoint, globus_client, parent=None):
        self.endpoint = endpoint
        super(GlobusFileView, self).__init__(globus_client, parent=parent)
        self.get_dir_contents('~')

    def refresh(self, path):
        self.path = path
        super(GlobusFileView, self).getDirContents(path, self.endpoint)
        self.pathChanged.emit(path)

    def downloadFile(self):
        fpath, outpath = super(GlobusFileView, self).downloadFile()
        method = self.client.transfer_generator
        desc = '{0} from {1}.'.format(os.path.split(fpath)[-1], self.endpoint)
        args = [self.endpoint, fpath, 'LOCAL ENDPOINT', outpath] #TODO Local endpoint again....
        kwargs = {}

        return desc, method, args, kwargs

    def transferFile(self):
        fpath, outpath = super(GlobusFileView, self).downloadFile()
        method = self.client.transfer_generator
        desc = '{0} from {1}.'.format(os.path.split(fpath)[-1], self.endpoint)
        args = [self.endpoint, fpath, 'DEST ENDPOINT', outpath] #TODO Dest endpoint geeet it....
        kwargs = {}

        return desc, method, args, kwargs

    def uploadFile(self):
        return None

    def deleteFile(self):
        fpath = self.getSelectedFilePath()
        self.client.delete_file(self.endpoint, fpath)

class SpotDatasetView(QtGui.QTreeWidget):
    """
    Tree widgets showing Spot datasets
    """

    def __init__(self, spot_client, parent=None):
        super(SpotDatasetView, self).__init__(parent)
        self.client = spot_client
        self.search_params = {'skipnum': '0', 'sortterm': 'fs.stage_date', 'sortterm': 'appmetadata.sdate',
                              'sorttype': 'desc', 'end_station': 'bl832'}
        self.getDatasets('')
        self.setHeaderHidden(True)

    def getDatasets(self, query):
        runnable = threads.RunnableMethod(self.createDatasetDictionary, self.client.search, query, **self.search_params)
        threads.queue.put(runnable)

    def createDatasetDictionary(self, data):
        tree_data = {}
        for index in range(len(data)):
            derived_data = {data[index]['fs']['stage']:
                            data[index]['name']}
            if 'derivatives' in data[index]['fs'].keys():
                derivatives = data[index]['fs']['derivatives']
                for d_index in range(len(derivatives)):
                    stage = derivatives[d_index]['dstage']
                    name = derivatives[d_index]['did'].split('/')[-1]
                    derived_data[stage] = name

            dataset = data[index]['fs']['dataset']
            tree_data[dataset] = derived_data
            self.fillTree(tree_data)

    def fillTree(self, data):
        self.clear()
        item = self.invisibleRootItem()
        self.addTreeItems(item, data)

    def addTreeItems(self, item, value):
        item.setExpanded(False)

        if type(value) is dict:
            for key, val in sorted(value.iteritems()):
                child = QtGui.QTreeWidgetItem()
                child.setText(0, unicode(key))
                item.addChild(child)
                self.addTreeItems(child, val)
        elif type(value) is list:
            for val in value:
                child = QtGui.QTreeWidgetItem()
                item.addChild(child)
                if type(val) is dict:
                    child.setText(0, '[dict]')
                    self.addTreeItems(child, val)
                elif type(val) is list:
                    child.setText(0, '[list]')
                    self.addTreeItems(child, val)
                else:
                    child.setText(0, unicode(val))
                child.setExpanded(False)
        else:
            child = QtGui.QTreeWidgetItem()
            child.setText(0, unicode(value))
            item.addChild(child)

    def getDatasetAndStage(self, file_name):
        dataset, stage, dset_item = None, None, None
        parent_items = self.findItems('*', QtCore.Qt.MatchWildcard)
        for item in parent_items:
            for child_item in [item.child(i) for i in range(item.childCount())]:
                if child_item.child(i).text(0) == file_name:
                    dset_item = child_item.child(i)

        if dset_item is not None:
            stage = str(dset_item.parent().text(0))
            dataset = str(dset_item.parent().parent().text(0))

        return stage, dataset

    def getSelectedFile(self):
        item = self.currentItem()
        fname = None
        if item is not None:
            if item.childCount() == 0:
                fname = item.text(0)
        return fname

    def downloadFile(self):
        fname = self.getSelectedFile()
        if fname is None:
            QtGui.QMessageBox.information(self, 'No file selected', 'Please select a file to download')
            return
        fileDialog = QtGui.QFileDialog(self, 'Save as', os.path.expanduser('~'))
        fileDialog.setAcceptMode(QtGui.QFileDialog.AcceptSave)
        fileDialog.selectFile(fname)
        if fileDialog.exec_():
            outpath = str(fileDialog.selectedFiles()[0])
            fpath, fname = os.path.split(outpath)
            desc = '{} from SPOT.'.format(fname)
            stage = self.currentItem().parent().text(0)
            dataset = self.currentItem().parent().parent().text(0)
            #stage, dataset = self.getDatasetAndStage(fname)
            method = self.client.download_dataset_generator
            args = [dataset, stage, fpath]
            kwargs = {'fname': fname}
            return desc, method, args, kwargs

    def transferFile(self):
        return None

    def uploadFile(self):
        return None

    def deleteFile(self):
        return None

class SpotDatasetExplorer(QtGui.QWidget):
    """
    Class that holds a SpotDatasetView and additional search boxes
    """

    def __init__(self, spot_client, parent=None):
        super(SpotDatasetExplorer, self).__init__(parent)

        self.file_view = SpotDatasetView(spot_client, self)
        self.search_box = QtGui.QLineEdit(self)
        self.sort_box = QtGui.QComboBox(self)
        self.order_box = QtGui.QComboBox(self)

        self.search_box.setPlaceholderText('Search Term')
        self.sort_box.addItems(['Beamline date', 'Processed date'])
        self.order_box.addItems(['Descending', 'Ascending'])

        l = QtGui.QVBoxLayout(self)
        l.setStretch(0, 0)
        l.setContentsMargins(0, 0, 0, 0)
        l.setSpacing(0)
        hl = QtGui.QHBoxLayout()
        hl.addWidget(self.sort_box)
        hl.addWidget(self.order_box)
        vl = QtGui.QVBoxLayout()
        vl.addWidget(self.search_box)
        vl.addLayout(hl)
        l.addLayout(vl)
        l.addWidget(self.file_view)

        self.search_box.returnPressed.connect(self.searchReturnPressed)

    def searchReturnPressed(self):
        if self.sort_box.currentText() == 'Beamline date':
            self.file_view.search_params['sortterm'] = 'appmetadata.sdate'
        else:
            self.file_view.search_params['sortterm'] = 'appmetadata.stage_date'

        if self.order_box.currentText() == 'Descending':
            self.file_view.search_params['sorttype'] = 'desc'
        else:
            self.file_view.search_params['sorttype'] = 'asc'
        query = str(self.search_box.text())

        self.file_view.getDatasets(query)

    def getSelectedFilePath(self):
        return os.path.join(*self.file_view.getDatasetAndStage())

    def getRawDatasetList(self):
        widget = self.file_view
        items = []

        parent_items = widget.findItems('*', QtCore.Qt.MatchWildcard)
        child_items = []
        for item in parent_items:
            child_items += [item.child(i) for i in range(item.childCount())]
        for item in child_items:
            items += [item.child(i).text(0) for i in range(item.childCount()) if item.text(0) == 'raw']

        return items


class TabBarPlus(QtGui.QTabBar):
    """
    Tab bar that has a plus button floating to the right of the tabs.
    """

    plusClicked = QtCore.Signal()

    def __init__(self, parent=None):
        super(TabBarPlus, self).__init__(parent)

        self.plus_button = QtGui.QPushButton(" + ")
        self.plus_button.setParent(self)
        self.plus_button.setMaximumSize(32, 32)
        self.plus_button.setMinimumSize(32, 32)
        self.plus_button.clicked.connect(self.plusClicked.emit)
        self.movePlusButton()  # Move to the correct location

    def sizeHint(self):
        sizeHint = QtGui.QTabBar.sizeHint(self)
        width = sizeHint.width()
        height = sizeHint.height()
        return QtCore.QSize(width+32, height)

    def resizeEvent(self, event):
        super(TabBarPlus, self).resizeEvent(event)
        self.movePlusButton()

    def tabLayoutChange(self):
        super(TabBarPlus, self).tabLayoutChange()
        self.movePlusButton()

    def movePlusButton(self):
        # Find the width of all of the tabs
        size = 0
        for i in range(self.count()):
            size += self.tabRect(i).width()

        h = self.geometry().top()
        w = self.width()
        if size > w:
            self.plus_button.move(w, h)
        else:
            self.plus_button.move(size, h)


class MultipleFileExplorer(QtGui.QTabWidget):
    """
    Class for multiple location file explorer, capability to add GlobusFileView Tabs
    """

    sigOpenDataset = QtCore.Signal(str, str)
    sigExternalJob = QtCore.Signal(str, str, object, list, dict)

    newtabmenu = QtGui.QMenu(None)
    loginspot = QtGui.QAction('Login to Spot',newtabmenu)
    loginedison = QtGui.QAction('Login to Edison',newtabmenu)
    logincori = QtGui.QAction('Login to Cori',newtabmenu)
    newtabmenu.addActions([loginspot,loginedison,logincori])

    def __init__(self, parent=None):
        super(MultipleFileExplorer, self).__init__(parent)
        self.nersc_login = False
        self.globus_login = False
        self.explorers = OrderedDict()

        self.tab = TabBarPlus()
        self.setTabBar(self.tab)

        self.setTabsClosable(True)

        self.explorers['Local'] = FileExplorer(LocalFileView(self), self)
        self.addFileExplorer('Local', self.explorers['Local'])

        self.tab.plusClicked.connect(self.onPlusClicked)
        self.tabCloseRequested.connect(self.removeTab)

        self.loginspot.triggered.connect(self.loginspotdialog)

    def addFileExplorer(self, name, file_explorer, closable=False):
        self.explorers[name] = file_explorer
        tab = self.addTab(file_explorer, name)
        if closable is False:
            try:
                self.tabBar().tabButton(tab, QtGui.QTabBar.RightSide).resize(0, 0)
                self.tabBar().tabButton(tab, QtGui.QTabBar.RightSide).hide()
            except AttributeError:
                self.tabBar().tabButton(tab, QtGui.QTabBar.LeftSide).resize(0, 0)
                self.tabBar().tabButton(tab, QtGui.QTabBar.LeftSide).hide()
        else:
            self.tabBar().tabButton(tab, QtGui.QTabBar.RightSide).resize(12, 12)

    def addNERSCTab(self, client, system, closable=False):
        explorer = FileExplorer(NERSCFileView(client, system, self))
        self.addFileExplorer(system.capitalize(), explorer, closable=closable)

    def loginspotdialog(self):
        ld=login.LoginDialog(remotename='SPOT',fileexplorer=self)
        ld.exec_()

    def addSPOTTab(self, client, closable=False):
        explorer = SpotDatasetExplorer(client, self)
        self.addFileExplorer('SPOT', explorer, closable=closable)
        self.nersc_login = True

    def addGlobusTab(self, endpoint, client, closable=True):
        explorer = FileExplorer(GlobusFileView(endpoint, client, self))
        name = endpoint.split('#')[-1]
        self.addFileExplorer(name, explorer, closable=closable)
        self.globus_login = True

    def removeTab(self, p_int):
        self.explorers.pop(self.explorers.keys()[p_int])
        self.widget(p_int).deleteLater()
        super(MultipleFileExplorer, self).removeTab(p_int)

    def onPlusClicked(self):
        self.newtabmenu.popup(QtGui.QCursor.pos())

    def getSelectedFilePath(self):
        return self.currentWidget().getSelectedFilePath()

    def getCurrentPath(self):
        return self.currentWidget().path

    def getPath(self, tab_name):
        return self.explorers[tab_name].path

    def openDataset(self):
        data_type, ok = QtGui.QInputDialog.getItem(self, 'Open Dataset', 'Data type:', ['raw', 'reconstruction'],
                                                   editable=False)
        if ok:
            path = self.getSelectedFilePath()
            self.sigOpenDataset.emit(path, data_type)

    def deleteFile(self):
        fname = self.currentWidget().file_view.getSelectedFile()
        if fname is None:
            return

        r = QtGui.QMessageBox.warning(self, 'Delete file', 'Are you sure you want to delete {}'.format(fname),
                                       QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)
        if r == QtGui.QMessageBox.Yes:
            self.currentWidget().file_view.deleteFile()

    def uploadFile(self):
        upload = self.currentWidget().file_view.uploadFile()
        if upload is not None:
            desc, method, args, kwargs = upload
            self.sigExternalJob.emit('Upload', desc, method, args, kwargs)

    def downloadFile(self):
        download = self.currentWidget().file_view.downloadFile()
        if download is not None:
            desc, method, args, kwargs = download
            self.sigExternalJob.emit('Download', desc, method, args,    kwargs)

    def transferFile(self):
        transfer = self.currentWidget().file_view.transferFile()
        if transfer is not None:
            desc, method, args, kwargs = transfer
            self.sigExternalJob.emit('Transfer', desc, method, args, kwargs)