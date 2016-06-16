# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 17:22:00 2015

@author: lbluque
"""

import os
import tempfile
from PySide import QtGui, QtCore
from collections import OrderedDict
from xicam import threads
from xicam import xglobals
from pipeline import loader,pathtools


NERSC_SYSTEMS = ['cori', 'edison']


class LocalFileView(QtGui.QTreeView):
    """
    Local file explorer tree view
    """

    pathChanged = QtCore.Signal(str)
    sigOpen = QtCore.Signal(list)
    sigOpenDir = QtCore.Signal(str)
    sigDelete = QtCore.Signal()

    def __init__(self, parent=None):
        super(LocalFileView, self).__init__(parent)

        self.file_model = QtGui.QFileSystemModel()
        self.setModel(self.file_model)
        self.path = pathtools.getRoot()
        self.refresh(self.path)

        header = self.header()
        self.setHeaderHidden(True)
        for i in range(1, 4):
            header.hideSection(i)
#        filefilter = map(lambda s: '*'+s,loader.acceptableexts)
#        self.file_model.setNameFilters(filefilter)
        self.file_model.setNameFilterDisables(False)
        self.file_model.setResolveSymlinks(True)
        self.expandAll()
        self.sortByColumn(0, QtCore.Qt.AscendingOrder)
        self.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        self.setSelectionMode(self.ExtendedSelection)
        self.setIconSize(QtCore.QSize(16, 16))

        self.menu = QtGui.QMenu()
        standardActions = [QtGui.QAction('Open', self), QtGui.QAction('Delete', self)]
        standardActions[0].triggered.connect(self.openActionTriggered)
        standardActions[1].triggered.connect(self.sigDelete.emit)
        self.menu.addActions(standardActions)

        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.menuActionClicked)

        self.doubleClicked.connect(self.onDoubleClick)

    def refresh(self, path=None):
        """
        Refresh the file tree, or switch directories and refresh
        """
        if path is None:
            path = self.path
        else:
            self.path = path

        root = QtCore.QDir(path)
        self.file_model.setRootPath(root.absolutePath())
        self.setRootIndex(self.file_model.index(root.absolutePath()))
        self.pathChanged.emit(path)

    def onDoubleClick(self, index):
        item = self.file_model.index(index.row(), 0, index.parent())
        path = self.file_model.filePath(item)

        if os.path.isdir(path):
            self.refresh(path=path)
        else:
            self.sigOpen.emit([path])

    def openActionTriggered(self):

        indices = self.selectedIndexes()
        paths = [self.file_model.filePath(index) for index in indices]

        if os.path.isdir(paths[0]):
            self.refresh(path=paths[0])
        else:
            self.sigOpen.emit(paths)

    def getSelectedFilePath(self):
        selected = str(self.file_model.filePath(self.currentIndex()))
        if selected == '':
            selected = None
        return selected

    def getSelectedFile(self):
        selected = str(self.file_model.fileName(self.currentIndex()))
        if selected == '':
            selected = None
        return selected

    def uploadFile(self):
        fpath = self.getSelectedFilePath()
        return None

    def deleteFile(self):
        self.file_model.remove(self.currentIndex())

    def openFile(self):
        print 'Open files here or emit open signal to whoever wants it'

    def menuActionClicked(self, position):
        self.menu.exec_(self.viewport().mapToGlobal(position))

    def addMenuAction(self, action_name, triggered_slot):
        action = QtGui.QAction(action_name, self)
        action.triggered.connect(triggered_slot)
        self.menu.addAction(action)


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
        icon.addPixmap(QtGui.QPixmap('gui/back.png'), QtGui.QIcon.Normal, QtGui.QIcon.Off)
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
        self.file_view.refresh(path=path)

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
    sigDelete = QtCore.Signal()
    sigOpen = QtCore.Signal(list)
    sigDownload = QtCore.Signal(tuple)
    sigTransfer = QtCore.Signal(tuple)

    def __init__(self, remote_client, parent=None):
        super(RemoteFileView, self).__init__(parent)
        self.path = None
        self.client = remote_client
        self.itemDoubleClicked.connect(self.onDoubleClick)

        self.menu = QtGui.QMenu()
        standardActions = [QtGui.QAction('Open', self), QtGui.QAction('Download', self)]
        # , QtGui.QAction('Delete', self), QtGui.QAction('Transfer', self)]
        standardActions[0].triggered.connect(lambda: self.onDoubleClick(self.currentItem()))
        # standardActions[1].triggered.connect(self.sigDelete.emit)
        standardActions[1].triggered.connect(self.sigDownload.emit)
        # standardActions[3].triggered.connect(self.sigTransfer.emit)
        self.menu.addActions(standardActions)
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.menuActionClicked)

    def onDoubleClick(self, item):
        file_name = item.text()

        if '.' in file_name:
            save_path = os.path.join(tempfile.gettempdir(), file_name)
            self.downloadFile(save_path=save_path, fslot=(lambda: self.sigOpen.emit([save_path])))
        elif len(file_name.split('.')) == 1:
            path = self.path + '/' + str(file_name)
            self.refresh(path=path)

    def getSelectedFilePath(self):
        return '{0}/{1}'.format(self.path, str(self.currentItem().text()))

    def getSelectedFile(self):
        return str(self.currentItem().text())

    def getDirContents(self, path, *args):
        runnable = threads.RunnableMethod(self.client.get_dir_contents, method_args=(path,) + args,
                                          callback_slot=self.fillList)
        threads.add_to_queue(runnable)

    def refresh(self, path=None):
        if path is None:
            path = self.path
        else:
            self.path = path
        self.pathChanged.emit(path)

    def fillList(self, value):
        self.clear()
        for item in value:
            if item['name'] != '.' and item['name'] != '..':
                self.addItem(item['name'])

    def downloadFile(self):
        fpath = self.getSelectedFilePath()
        return fpath

    def transferFile(self):
        return None

    def deleteFile(self):
        return None

    def menuActionClicked(self, position):
        self.menu.exec_(self.viewport().mapToGlobal(position))


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

    def refresh(self, path=None):
        if path is None:
            path = self.path

        super(NERSCFileView, self).getDirContents(path, self.system)
        super(NERSCFileView, self).refresh(path=path)

    def downloadFile(self, save_path=None, fslot=None):
        fpath = super(NERSCFileView, self).downloadFile()
        size = self.client.get_file_size(fpath, self.system)
        desc = 'File {0} from {1}'.format(fpath, self.system)
        if size < 100*2**20:
            method = self.client.download_file_generator
            args = [fpath, self.system]
            kwargs = {}
            if save_path is not None:
                kwargs['save_path'] = save_path
        else:
            # USE GLOBUS API CALL HERE, Think of how to get access to client
            pass

        self.sigDownload.emit((desc, method, args, kwargs, fslot))

    def transferFile(self):
        return None

    def deleteFile(self):
        runnable = threads.RunnableMethod(self.client.delete_file,
                                          method_args=(self.getSelectedFilePath(), self.system),
                                          finished_slot=self.refresh)
        threads.add_to_queue(runnable)


class GlobusFileView(RemoteFileView):
    """
    File explorer for Globus endpoints
    """

    def __init__(self, endpoint, globus_client, parent=None):
        self.endpoint = endpoint
        super(GlobusFileView, self).__init__(globus_client, parent=parent)
        self.get_dir_contents('~')

    def refresh(self, path=None):
        if path is None:
            path = self.path
        super(GlobusFileView, self).getDirContents(path, self.endpoint)
        super(GlobusFileView, self).refresh(path=path)

    def downloadFile(self, save_path=None, fslot=None):
        fpath = super(GlobusFileView, self).downloadFile()
        method = self.client.transfer_generator
        desc = '{0} from {1}.'.format(os.path.split(fpath)[-1], self.endpoint)
        args = [self.endpoint, fpath, 'LOCAL ENDPOINT'] #TODO Local endpoint again....
        kwargs = {}

        self.sigDownload.emit((desc, method, args, kwargs, fslot))

    def transferFile(self):
        fpath, outpath = super(GlobusFileView, self).downloadFile()
        method = self.client.transfer_generator
        desc = '{0} from {1}.'.format(os.path.split(fpath)[-1], self.endpoint)
        args = [self.endpoint, fpath, 'DEST ENDPOINT', outpath] #TODO Dest endpoint geeet it....
        kwargs = {}
        return desc, method, args, kwargs

    def deleteFile(self):
        fpath = self.getSelectedFilePath()
        self.client.delete_file(self.endpoint, fpath)


class SpotDatasetView(QtGui.QTreeWidget):
    """
    Tree widgets showing Spot datasets
    """

    sigOpen = QtCore.Signal(list)
    sigDownload = QtCore.Signal(tuple)
    sigTransfer = QtCore.Signal()


    def __init__(self, spot_client, parent=None):
        super(SpotDatasetView, self).__init__(parent)
        self.client = spot_client
        self.search_params = {'skipnum': '0', 'sortterm': 'fs.stage_date', 'sortterm': 'appmetadata.sdate',
                              'sorttype': 'desc', 'end_station': 'bl832'}
        self.getDatasets('')
        self.setHeaderHidden(True)

        self.menu = QtGui.QMenu()
        standardActions = [QtGui.QAction('Open', self), QtGui.QAction('Download', self)]
        #, QtGui.QAction('Preview', self), QtGui.QAction('Transfer', self)]
        standardActions[0].triggered.connect(lambda: self.onDoubleClick(self.currentItem()))
        # standardActions[1].triggered.connect(self.previewDataset)
        standardActions[1].triggered.connect(self.downloadFile)
        # standardActions[3].triggered.connect(self.sigTransfer.emit)
        self.menu.addActions(standardActions)
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.menuActionClicked)
        self.itemDoubleClicked.connect(self.onDoubleClick)

    def onDoubleClick(self, item):
        if item.childCount() != 0:
            return
        file_name = item.text(0)
        save_path = os.path.join(tempfile.gettempdir(), file_name)
        self.downloadFile(save_path=save_path, fslot=(lambda: self.sigOpen.emit([save_path])))

    def getDatasets(self, query):
        runnable = threads.RunnableMethod(self.client.search, method_args=(query, ), method_kwargs=self.search_params,
                                          callback_slot=self.createDatasetDictionary)
        threads.add_to_queue(runnable)

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

    def getDatasetAndStage(self):
        stage = self.currentItem().parent().text(0)
        dataset = self.currentItem().parent().parent().text(0)
        # dataset, stage, dset_item = None, None, None
        # parent_items = self.findItems('*', QtCore.Qt.MatchWildcard)
        # for item in parent_items:
        #     for child_item in [item.child(i) for i in range(item.childCount())]:
        #         for child_child_item in [child_item.child(j) for j in range(child_item.childCount())]:
        #             if child_child_item.text(0) == file_name:
        #                 dset_item = child_child_item
        #
        # if dset_item is not None:
        #     stage = str(dset_item.parent().text(0))
        #     dataset = str(dset_item.parent().parent().text(0))

        return stage, dataset

    def getSelectedFile(self):
        item = self.currentItem()
        fname = None
        if item is not None:
            if item.childCount() == 0:
                fname = item.text(0)
        return fname

    def downloadFile(self, save_path=None, fslot=None):
        dset = self.getSelectedFile()

        if dset is not None:
            stage, dataset = self.getDatasetAndStage()
            desc = '{} from SPOT.'.format(dset)
            method = self.client.download_dataset_generator
            args = [dataset, stage]
            kwargs = {}
            if save_path is not None:
                kwargs['save_path'] = save_path

            self.sigDownload.emit((desc, method, args, kwargs, fslot))

    def transferFile(self):
        fname = self.getSelectedFile()
        stage, dataset = self.getDatasetAndStage()
        system = self.client.system
        path = self.client.scratch_dir
        desc = '{0} transfer from spot to {1}.'.format(fname, system)
        method = self.client.transfer_2_nersc
        args = [dataset, stage, path, system]
        kwargs = {}

        return desc, method, args, kwargs

    def deleteFile(self):
        return None

    def previewDataset(self):
        print "Not implemented!"

    def menuActionClicked(self, position):
        self.menu.exec_(self.viewport().mapToGlobal(position))


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
        self.setDocumentMode(True)

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

    sigLoginSuccess = QtCore.Signal(bool)
    sigLoginRequest = QtCore.Signal()
    sigProgJob = QtCore.Signal(str, object, list, dict, object)
    sigPulsJob = QtCore.Signal(str, object, list, dict, object)
    sigOpenFiles = QtCore.Signal(list)

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
        self.explorers['Local'].file_view.sigDelete.connect(self.deleteFile)
        self.explorers['Local'].file_view.sigOpen.connect(self.openFiles)

        self.jobtab = JobTable(self)
        # Do not understand why I need to add it and remove it so that its not added as a seperate widget
        self.addTab(self.jobtab, 'Jobs')
        self.removeTab(1)

        self.sigProgJob.connect(self.jobtab.addProgJob)
        self.sigPulsJob.connect(self.jobtab.addPulseJob)

        self.tab.plusClicked.connect(self.onPlusClicked)
        self.tabCloseRequested.connect(self.removeTab)

        self.newtabmenu = QtGui.QMenu(None)
        self.addspot = QtGui.QAction('SPOT', self.newtabmenu)
        self.addcori = QtGui.QAction('Cori', self.newtabmenu)
        self.addedison = QtGui.QAction('Edison', self.newtabmenu)
        self.addglobus = QtGui.QAction('Globus endpoint', self.newtabmenu)
        self.standard_actions = [self.addspot, self.addcori, self.addedison] #, self.addglobus]
        self.newtabmenu.addActions(self.standard_actions)
        self.addspot.triggered.connect(self.addSPOTTab)
        self.addedison.triggered.connect(lambda: self.addNERSCTab('edison'))
        self.addcori.triggered.connect(lambda: self.addNERSCTab('cori'))

    def enableActions(self):
        for action in self.standard_actions:
            action.setEnabled(True)

    def addFileExplorer(self, name, file_explorer, closable=False):
        self.explorers[name] = file_explorer
        idx = len(self.explorers) - 1
        tab = self.insertTab(idx, file_explorer, name)
        if closable is False:
            try:
                self.tabBar().tabButton(tab, QtGui.QTabBar.RightSide).resize(0, 0)
                self.tabBar().tabButton(tab, QtGui.QTabBar.RightSide).hide()
            except AttributeError:
                self.tabBar().tabButton(tab, QtGui.QTabBar.LeftSide).resize(0, 0)
                self.tabBar().tabButton(tab, QtGui.QTabBar.LeftSide).hide()

    @xglobals.login_exeption_handle
    def addNERSCTab(self, system, closable=True, *args):
        self.sigLoginSuccess.emit(xglobals.spot_client.logged_in)
        self.sigLoginRequest.emit()
        client = xglobals.load_spot(self.addNERSCTab, system, closable)
        explorer = FileExplorer(NERSCFileView(client, system, self))
        explorer.file_view.sigOpen.connect(self.openFiles)
        explorer.file_view.sigDelete.connect(self.deleteFile)
        explorer.file_view.sigDownload.connect(self.downloadFile)
        explorer.file_view.sigTransfer.connect(self.transferFile)
        self.addFileExplorer(system.capitalize(), explorer, closable=closable)
        action = self.addcori if system == 'cori' else self.addedison
        action.setEnabled(False)

    @xglobals.login_exeption_handle
    def addSPOTTab(self, closable=True, *args):
        self.sigLoginSuccess.emit(xglobals.spot_client.logged_in)
        self.sigLoginRequest.emit()
        client = xglobals.load_spot(self.addSPOTTab, closable)
        explorer = SpotDatasetExplorer(client, self)
        explorer.file_view.sigOpen.connect(self.openFiles)
        explorer.file_view.sigDownload.connect(self.downloadFile)
        explorer.file_view.sigTransfer.connect(self.transferFile)
        self.addFileExplorer('SPOT', explorer, closable=closable)
        self.addspot.setEnabled(False)

    @xglobals.login_exeption_handle
    def addGlobusTab(self, endpoint, closable=True, *args):
        self.sigLoginSuccess.emit(xglobals.globus_client.logged_in)
        self.sigLoginRequest.emit()
        client = xglobals.load_spot(self.addGlobusTab, endpoint, closable)
        explorer = FileExplorer(GlobusFileView(endpoint, client, self))
        name = endpoint.split('#')[-1]
        self.addFileExplorer(name, explorer, closable=closable)

    def removeTab(self, p_int):
        if self.tabText(p_int) != 'Jobs':
            name = self.explorers.keys()[p_int]
            self.explorers.pop(name)
            self.widget(p_int).deleteLater()
            action = [action for action in self.standard_actions if action.text() == name][0]
            action.setEnabled(True)
        super(MultipleFileExplorer, self).removeTab(p_int)

    def removeTabs(self):
        for i in xrange(1, self.count()):
            self.removeTab(1)

    def onPlusClicked(self):
        self.newtabmenu.popup(QtGui.QCursor.pos())

    def getSelectedFilePath(self):
        return self.currentWidget().getSelectedFilePath()

    def getCurrentPath(self):
        return self.currentWidget().path

    def getPath(self, tab_name):
        return self.explorers[tab_name].path

    def openFiles(self, paths):
        if len(paths) > 0:
            self.sigOpenFiles.emit(paths)

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
            self.sigProgJob.emit(desc, method, args, kwargs)
            self.addTab(self.jobtab, 'Jobs')

    def downloadFile(self, download):
        if download is not None:
            desc, method, args, kwargs, fslot = download
            if 'save_path' not in kwargs:
                fileDialog = QtGui.QFileDialog(self, 'Save as', os.path.expanduser('~'))
                fileDialog.setAcceptMode(QtGui.QFileDialog.AcceptSave)
                fileDialog.selectFile(self.currentWidget().file_view.getSelectedFile())
                if fileDialog.exec_():
                    kwargs['save_path'] = str(fileDialog.selectedFiles()[0])
                else:
                    return

            self.sigProgJob.emit(desc, method, args, kwargs, fslot)
            self.addTab(self.jobtab, 'Jobs')

    def transferFile(self):
        transfer = self.currentWidget().file_view.transferFile()
        if transfer is not None:
            desc, method, args, kwargs = transfer
            if isinstance(self.currentWidget().file_view, SpotDatasetView):
                # TODO Find a way to track the progress of rsync or cp on nersc to run the job as an iterator and get rid of if statement
                print 'This absurd emit'
                self.sigPulsJob.emit(desc, method, args, kwargs)
            else:
                self.sigProgJob.emit(desc, method, args, kwargs)
            self.addTab(self.jobtab, 'Jobs')


class JobTable(QtGui.QTableWidget):
    """
    Class with table of download, upload and transfer jobs
    """

    def __init__(self, parent=None):
        super(JobTable, self).__init__(0, 2, parent=parent)
        self.jobs = []

        self.setHorizontalHeaderLabels(['Progress', 'Description'])
        self.setFrameShape(QtGui.QFrame.NoFrame)
        self.verticalHeader().hide()
        self.horizontalHeader().setClickable(False)
        self.horizontalHeader().setStretchLastSection(True)
        self.horizontalHeader().setDefaultAlignment(QtCore.Qt.AlignLeft)
        self.setShowGrid(False)

    def addJob(self, job_desc):
        row_num = self.rowCount()
        self.insertRow(row_num)
        jobentry = JobEntry()
        jobentry.setDescription(job_desc)
        self.jobs.append(jobentry)
        self.setCellWidget(row_num, 0, jobentry.widget)
        self.setCellWidget(row_num, 1, jobentry.desc_label)
        return jobentry

    @QtCore.Slot(str, object, list, dict)
    def addProgJob(self, job_desc, generator, args, kwargs, finish_slot=None):
        job_entry = self.addJob(job_desc) #TODO add interrupt signal to job entries!
        runnable = threads.RunnableIterator(generator, generator_args=args, generator_kwargs=kwargs,
                                            callback_slot=job_entry.progress, finished_slot=finish_slot)
        threads.add_to_queue(runnable)

    @QtCore.Slot(str, object, list, dict)
    def addPulseJob(self, job_type, job_desc, method, args, kwargs):
        job_entry = self.addJob(job_type, job_desc)
        job_entry.pulseStart()
        runnable = threads.RunnableMethod(method, method_args=args, method_kwargs=kwargs,
                                          finished_slot=job_entry.pulseStop)
        threads.add_to_queue(runnable)

    def removeJob(self, jobentry):
        idx = self.jobs.index(jobentry)
        del self.jobs[idx]
        self.removeRow(idx)
        jobentry.deleteLater()


class JobEntry(QtGui.QWidget):
    """
    Job entries
    """

    sigCancel = QtCore.Signal(object)

    def __init__(self):
        super(JobEntry, self).__init__()
        self.desc_label = QtGui.QLabel()
        self.desc_label.setAlignment(QtCore.Qt.AlignVCenter)
        self.progressbar = QtGui.QProgressBar()
        self.progressbar.setRange(0, 100)
        self.widget = QtGui.QWidget()
        self.widget.setLayout(QtGui.QHBoxLayout())
        self.widget.layout().setContentsMargins(0, 0, 0, 0)
        self.widget.layout().addWidget(self.progressbar)

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
        self.progressbar.setRange(0, 100)
        self.progress(1)