# -*- coding: utf-8 -*-

__author__ = "Luis Barroso-Luque"
__copyright__ = "Copyright 2016, CAMERA, LBL, ALS"
__credits__ = ["Ronald J Pandolfi", "Dinesh Kumar", "Singanallur Venkatakrishnan", "Luis Luque", "Alexander Hexemer"]
__license__ = ""
__version__ = "1.2.1"
__maintainer__ = "Ronald J Pandolfi"
__email__ = "ronpandolfi@lbl.gov"
__status__ = "Beta"


import os
import tempfile
from functools import partial
from PySide import QtGui, QtCore
from collections import OrderedDict
from xicam import threads
from xicam import clientmanager as cmanager
from pipeline import pathtools, msg
from xicam import config
from modpkgs import guiinvoker

class LocalFileView(QtGui.QTreeView):
    """
    Local file explorer tree view
    """

    pathChanged = QtCore.Signal(str)
    sigOpen = QtCore.Signal(list)
    sigOpenFolder = QtCore.Signal(list)
    sigDelete = QtCore.Signal(list)
    sigUpload = QtCore.Signal(list)
    sigItemPreview = QtCore.Signal(str)

    def __init__(self, parent=None):
        super(LocalFileView, self).__init__(parent)

        self.file_model = QtGui.QFileSystemModel()
        self.setModel(self.file_model)
        self.path = config.settings['defaultlocalpath']
        if not self.path: self.path = os.path.expanduser('~')  # pathtools.getRoot()
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
        standardActions = [QtGui.QAction('Open', self), QtGui.QAction('Open Folder', self),
                           QtGui.QAction('Delete', self)]
        standardActions[0].triggered.connect(self.handleOpenAction)
        standardActions[1].triggered.connect(self.handleOpenFolderAction)
        standardActions[2].triggered.connect(self.handleDeleteAction)
        self.menu.addActions(standardActions)

        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.menuRequested)

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
        config.settings['defaultlocalpath'] = path

    def menuRequested(self, position):
        self.menu.exec_(self.viewport().mapToGlobal(position))

    def onDoubleClick(self, index):
        path = self.file_model.filePath(index)
        if os.path.isdir(path):
            self.refresh(path=path)
        else:
            self.sigOpen.emit([path])

    def currentChanged(self, current, previous):
        path = self.file_model.filePath(current)
        if os.path.isfile(path):
            self.sigItemPreview.emit(path)

    def getSelectedFilePaths(self):
        paths = [self.file_model.filePath(index) for index in self.selectedIndexes()]
        return paths

    def getSelectedFile(self):
        selected = str(self.file_model.fileName(self.currentIndex()))
        if selected == '':
            selected = None
        return selected

    def deleteSelection(self):
        for index in self.selectedIndexes():
            self.file_model.remove(index)

    def handleOpenAction(self):
        paths = self.getSelectedFilePaths()
        if os.path.isdir(paths[0]) and len(paths) == 1:
            self.refresh(path=paths[0])
        else:
            self.sigOpen.emit(paths)

    def handleOpenFolderAction(self):
        paths = self.getSelectedFilePaths()
        if os.path.isdir(paths[0]) and len(paths) == 1:
            self.sigOpenFolder.emit(paths)
        else:
            pass

    def handleDeleteAction(self):
        paths = self.getSelectedFilePaths()
        self.sigDelete.emit(paths)

    def handleUploadAction(self):
        paths = self.getSelectedFilePaths()
        self.sigUpload.emit(paths)


class RemoteFileView(QtGui.QListWidget):
    """
    Remote file explorer for REST API clients(NERSC and Globus)
    """
    pathChanged = QtCore.Signal(str)
    sigDelete = QtCore.Signal(list)
    sigOpen = QtCore.Signal(list)
    sigOpenFolder = QtCore.Signal(list)
    sigDownload = QtCore.Signal(str, str, object, tuple, dict, object)
    sigTransfer = QtCore.Signal(str, str, object, tuple, dict, object)
    sigItemPreview = QtCore.Signal(str)

    def __init__(self, remote_client, parent=None):
        super(RemoteFileView, self).__init__(parent)
        self.path = None
        self.client = remote_client
        self.itemDoubleClicked.connect(self.onDoubleClick)

        self.menu = QtGui.QMenu()
        standardActions = [QtGui.QAction('Open', self), QtGui.QAction('Open Folder', self). QtGui.QAction('Download', self),
                           QtGui.QAction('Delete', self), QtGui.QAction('Transfer', self)]
        standardActions[0].triggered.connect(self.handleOpenAction)
        standardActions[0].triggered.connect(self.handleOpenFolderAction)
        standardActions[2].triggered.connect(self.handleDownloadAction)
        standardActions[3].triggered.connect(self.handleDeleteAction)
        standardActions[4].triggered.connect(self.handleTransferAction)
        self.menu.addActions(standardActions)
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.menuRequested)

    def refresh(self, path=None):
        if path is None:
            path = self.path
        else:
            self.path = path
        self.pathChanged.emit(path)

    def menuRequested(self, position):
        self.menu.exec_(self.viewport().mapToGlobal(position))

    def onDoubleClick(self, item):
        file_name = item.text(0)
        if len(file_name.split('.')) == 1:
            path = self.path + '/' + str(file_name)
            self.refresh(path=path)

    def currentChanged(self, current, previous):
        pass

    def getSelectedFilePaths(self):
        paths = ['{0}/{1}'.format(self.path, item.text()) for item in self.selectedItems()]
        return paths

    def getSelectedFile(self):
        return str(self.currentItem().text())

    @threads.method(callback_slot=lambda self: self.fillList)
    def getDirContents(self, path, *args):
        self.client.get_dir_contents(path, *args)

    def fillList(self, value):
        self.clear()
        for item in value:
            if item['name'] != '.' and item['name'] != '..':
                self.addItem(item['name'])

    def handleOpenAction(self):
        pass

    def handleOpenFolderAction(self):
        pass

    def handleTransferAction(self):
        pass

    def handleDownloadAction(self):
        pass

    def handleDeleteAction(self):
        pass


# This has been replaced with SFTP based client see SFTPFileView
# class NERSCFileView(RemoteFileView):
# """
#     File explorer for NERSC systems, must be passed a client and a worker to make REST calls
#     """
#
#     def __init__(self, nersc_client, system, parent=None):
#         self.system = system
#         super(NERSCFileView, self).__init__(nersc_client, parent=parent)
#         self.client.set_scratch_dir(system)
#         self.path = self.client.scratch_dir
#         self.getDirContents(self.path, self.system)
#
#     def refresh(self, path=None):
#         if path is None:
#             path = self.path
#
#         super(NERSCFileView, self).getDirContents(path, self.system)
#         super(NERSCFileView, self).refresh(path=path)
#
#    def onDoubleClick(self, item):
#        file_name = item.text()
#        if '.' in file_name:
#            save_path = os.path.join(tempfile.gettempdir(), file_name)
#            self.handleDownloadAction(save_path=save_path, fslot=(lambda: self.sigOpen.emit([save_path])))
#        super(NERSCFileView, self).onDoubleClick(item)
#
#     def handleOpenAction(self):
#         paths = self.getSelectedFilePaths()
#         files = [os.path.split(path)[-1] for path in paths]
#         for file_name in files:
#             save_path = os.path.join(tempfile.gettempdir(), file_name)
#             self.handleDownloadAction(save_path=save_path, fslot=(lambda: self.sigOpen.emit([save_path])))
#
#     def handleDownloadAction(self, save_path=None, fslot=None):
#         fpath = super(NERSCFileView, self).handleDownloadAction()
#         size = self.client.get_file_size(fpath, self.system)
#         desc = 'File {0} from {1}'.format(fpath, self.system)
#         if size < 100*2**20:
#             method = self.client.download_file_generator
#             args = [fpath, self.system]
#             kwargs = {}
#             if save_path is not None:
#                 kwargs['save_path'] = save_path
#         else:
#             # USE GLOBUS API CALL HERE, Think of how to get access to client
#             pass
#
#         self.sigDownload.emit(desc, method, args, kwargs, fslot)
#
#     def handleTransferAction(self):
#         return None
#
#     def handleDeleteAction(self):
#         runnable = threads.RunnableMethod(self.client.delete_file,
#                                           method_args=(self.getSelectedFilePath(), self.system),
#                                           finished_slot=self.refresh)
#         threads.add_to_queue(runnable)


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

    def deleteSelection(self):
        paths = self.getSelectedFilePaths()
        for path in paths:
            self.client.delete_file(self.endpoint, path)

    def handleDownloadAction(self, save_path=None, fslot=None):
        paths = super(GlobusFileView, self).getSelectedFilePaths()
        for path in paths:
            name = os.path.split(path)[-1]
            method = self.client.transfer_generator
            desc = '{0} from {1}.'.format(name, self.endpoint)
            args = [self.endpoint, path, 'LOCAL ENDPOINT']  # TODO Local endpoint again....
            kwargs = {}
            self.sigDownload.emit(name, desc, method, args, kwargs, fslot)

    def handleTransferAction(self):
        # TODO implement this
        paths = super(GlobusFileView, self).getSelectedFilePaths()
        for path in paths:
            method = self.client.transfer_generator
            desc = '{0} from {1}.'.format(os.path.split(path)[-1], self.endpoint)
            args = [self.endpoint, path, 'DEST ENDPOINT']  # TODO Dest endpoint geeet it....
            kwargs = {}
            return desc, method, args, kwargs

    def handleDeleteAction(self):
        paths = self.getSelectedFilePaths()
        self.sigDelete.emit(paths)


class SFTPFileView(QtGui.QTreeWidget):
    pathChanged = QtCore.Signal(str)
    sigAddTopLevelItem = QtCore.Signal(str, str)
    sigDelete = QtCore.Signal(list)
    sigOpen = QtCore.Signal(list)
    sigOpenFolder = QtCore.Signal(list)
    sigDownload = QtCore.Signal(str, str, object, tuple, dict, object)
    sigTransfer = QtCore.Signal(str, str, object, tuple, dict, object)
    sigItemPreview = QtCore.Signal(str)

    def __init__(self, sftp_client, parent=None):
        super(SFTPFileView, self).__init__(parent=parent)
        self.client = sftp_client
        self.path = sftp_client.pwd

        self.sigAddTopLevelItem.connect(self.addTopLevelItem)

        self.itemExpanded.connect(self.getItemChildren)
        self.itemDoubleClicked.connect(self.onDoubleClick)
        self.setHeaderHidden(True)
        self.refresh()

        self.menu = QtGui.QMenu(parent=self)
        openAction = QtGui.QAction('Open', self)
        openFolderAction = QtGui.QAction('Open Folder', self)
        downloadAction = QtGui.QAction('Download', self)
        deleteAction = QtGui.QAction('Delete', self)
        openAction.triggered.connect(self.handleOpenAction)
        downloadAction.triggered.connect(self.handleDownloadAction)
        deleteAction.triggered.connect(self.handleDeleteAction)
        self.menu.addActions([openAction, openFolderAction, downloadAction, deleteAction])

        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.menuRequested)

    @threads.method(callback_slot=lambda self: self.pathChanged.emit(self.path))
    def refresh(self, path=None):
        guiinvoker.invoke_in_main_thread(self.clear)
        if path is not None:
            self.path = path
            self.client.cd(path)
        self.client.walktree(self.path, lambda x: self.sigAddTopLevelItem.emit(x, 'file'),
                             lambda x: self.sigAddTopLevelItem.emit(x, 'dir'), lambda: None, recurse=False)
        return self
        # runnable = threads.RunnableMethod(self.client.walktree,
        #                                   method_args=(self.path,
        #                                                lambda x: self.sigAddTopLevelItem.emit(x, 'file'),
        #                                                lambda x: self.sigAddTopLevelItem.emit(x, 'dir'),
        #                                                lambda: None),
        #                                   method_kwargs={'recurse': False},
        #                                   finished_slot=lambda: self.pathChanged.emit(self.path))
        # threads.add_to_queue(runnable)
        # Or on gui thread
        # self.client.walktree(self.path,
        # lambda x: self.createTopLevelItem(x, 'file'),
        #                      lambda x: self.createTopLevelItem(x, 'dir'),
        #                      lambda x: x, recurse=False)
        # self.pathChanged.emit(self.path)

    def menuRequested(self, position):
        self.menu.exec_(self.viewport().mapToGlobal(position))

    def addTopLevelItem(self, path, type):
        name = os.path.split(path)[-1]
        if type == 'file':
            icon = QtGui.QFileIconProvider().icon(QtGui.QFileIconProvider.File)
            item = SFTPFileTreeItem(name, path, icon, self)
        elif type == 'dir':
            name = os.path.split(path)[-1]
            icon = QtGui.QFileIconProvider().icon(QtGui.QFileIconProvider.Folder)
            item = SFTPDirTreeItem(name, self.client, path, icon, self)
        else:
            return
        super(SFTPFileView, self).addTopLevelItem(item)

    @staticmethod
    def isDir(item):
        if isinstance(item, SFTPDirTreeItem):
            return True
        else:
            return False

    def onDoubleClick(self, item):
        if self.isDir(item):
            self.path = item.path
            self.refresh()
        else:
            file_name = item.text(0)
            save_path = [os.path.join(tempfile.gettempdir(), file_name)]
            self.handleDownloadAction(save_paths=save_path, fslot=(lambda: self.sigOpen.emit(save_path)))

    def getItemChildren(self, item):
        if item.childCount() == 0:
            item.getChildren()

    def currentChanged(self, current, previous):
        pass

    def getSelectedFilePaths(self):
        paths = [item.path for item in self.selectedItems()]
        return paths

    def deleteSelection(self):
        paths = self.getSelectedFilePaths()
        for path in paths:
            self.client.remove(path)
        self.refresh()

    def handleOpenAction(self):
        paths = self.getSelectedFilePaths()
        save_paths = [os.path.join(tempfile.gettempdir(), os.path.split(path)[-1]) for path in paths]
        for save_path in save_paths:
            self.handleDownloadAction(save_paths=[save_path], fslot=(lambda: self.sigOpen.emit([save_path])))

    def handleDownloadAction(self, save_paths=None, fslot=None):
        paths = self.getSelectedFilePaths()
        items = self.selectedItems()
        if save_paths is None:
            save_paths = len(paths) * (None,)
        for path, item, save_path in zip(paths, items, save_paths):
            name = os.path.split(path)[-1]
            desc = '{0} from {1}'.format(name, self.client.host)
            args = (path,)
            kwargs = {}
            if save_path is not None:
                kwargs['localpath'] = save_path
            if self.isDir(item):
                method = self.client.get_r
            else:
                method = self.client.get
            self.sigDownload.emit(name, desc, method, args, kwargs, fslot)

    def handleDeleteAction(self):
        paths = self.getSelectedFilePaths()
        self.sigDelete.emit(paths)


class SpotDatasetView(QtGui.QTreeWidget):
    """
    Tree widgets showing Spot datasets as a file directory
    """

    sigOpen = QtCore.Signal(list)
    sigOpenFolder = QtCore.Signal(list)
    sigDownload = QtCore.Signal(str, str, object, tuple, dict, object)
    sigTransfer = QtCore.Signal(str, str, object, tuple, dict, object)
    sigItemPreview = QtCore.Signal(object)


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
        standardActions[0].triggered.connect(self.handleOpenAction)
        standardActions[1].triggered.connect(self.handleDownloadAction)
        # standardActions[2].triggered.connect(self.handlePreviewAction)
        # standardActions[3].triggered.connect(self.handleTransferAction)
        self.menu.addActions(standardActions)
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.menuRequested)
        self.itemDoubleClicked.connect(self.onDoubleClick)

    def menuRequested(self, position):
        if self.currentItem().childCount() != 0:
            return
        self.menu.exec_(self.viewport().mapToGlobal(position))

    def onDoubleClick(self, item):
        if item.childCount() != 0:
            return
        file_name = item.text(0)
        save_path = [os.path.join(tempfile.gettempdir(), file_name)]
        self.handleDownloadAction(save_paths=save_path, fslot=(lambda: self.sigOpen.emit(save_path)))

    def getDatasets(self, query):
        msg.showMessage('Searching SPOT database...')
        search = threads.method(callback_slot=self.createDatasetDictionary,
                                finished_slot=msg.clearMessage)(self.client.search)
        search(query, **self.search_params)

    @QtCore.Slot(dict)
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
                icon = QtGui.QFileIconProvider().icon(QtGui.QFileIconProvider.Folder)
                child = QtGui.QTreeWidgetItem([key], parent=self)
                child.setIcon(0, icon)
                item.addChild(child)
                self.addTreeItems(child, val)
        else:
            icon = QtGui.QFileIconProvider().icon(QtGui.QFileIconProvider.File)
            child = QtGui.QTreeWidgetItem([value], parent=self)
            child.setIcon(0, icon)
            item.addChild(child)

    def currentChanged(self, current, previous):
        item = self.itemFromIndex(current)
        try:
            if item.childCount() == 0:
                msg.showMessage('Loading preview...')
                dataset = item.parent().parent().text(0)
                stage = item.parent().text(0)
                bg_get_preview = threads.method(callback_slot=self.sigItemPreview.emit,
                                                except_slot=self.handleException)(self.client.get_image_as)
                bg_get_preview(dataset, stage, index=0)
        except AttributeError:
            pass

    def getStagesAndDatasets(self):
        dsets_stages = [(item.parent().parent().text(0), item.parent().text(0))
                        for item in self.selectedItems() if item.childCount() == 0]
        return dsets_stages

    def getSelectedDatasets(self):
        datasets = [item.text(0) for item in self.selectedItems() if item.childCount() == 0]
        return datasets

    def handleOpenAction(self):
        save_paths = map(lambda dset: os.path.join(tempfile.gettempdir(), dset), self.getSelectedDatasets())
        for path in save_paths:
            self.handleDownloadAction(save_paths=[path], fslot=(lambda: self.sigOpen.emit([path])))

    def handleDownloadAction(self, save_paths=None, fslot=None):
        names = self.getSelectedDatasets()
        stages_dsets = self.getStagesAndDatasets()
        if save_paths is None:
            save_paths = len(names) * (None, )
        for name, stage_dset, save_path in zip(names, stages_dsets, save_paths):
            desc = '{} from SPOT.'.format(name)
            method = self.client.download_dataset_generator
            args = stage_dset
            kwargs = {}
            if save_path is not None:
                kwargs['save_path'] = save_path
            self.sigDownload.emit(name, desc, method, args, kwargs, fslot)

    def handleTransferAction(self):
        # TODO need to implement this
        fname = self.getSelectedDatasets()
        stage, dataset = self.getStagesAndDatasets()
        system = self.client.system
        path = self.client.scratch_dir
        desc = '{0} transfer from spot to {1}.'.format(fname, system)
        method = self.client.transfer_2_nersc
        args = [dataset, stage, path, system]
        kwargs = {}
        return desc, method, args, kwargs

    def handleException(self, ex, tb):
        msg.showMessage('Unable to fetch preview from SPOT.')
        msg.logMessage(ex, level=40)


class FileExplorer(QtGui.QWidget):
    """
    Container for local and remote file tree view that adds back button and path label
    """

    def __init__(self, file_view, parent=None):
        super(FileExplorer, self).__init__(parent)

        self.file_view = file_view
        self.path_label = QtGui.QLineEdit(self)
        self.back_button = QtGui.QToolButton(self)
        self.back_button.setToolTip('Back')
        self.refresh_button = QtGui.QToolButton(self)
        self.refresh_button.setToolTip('Refresh')

        for button, icon_file in zip((self.back_button, self.refresh_button),
                                     ('xicam/gui/icons_44.png', 'xicam/gui/icons_57.png')):
            icon = QtGui.QIcon()
            icon.addPixmap(QtGui.QPixmap(icon_file), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            button.setIcon(icon)
            button.setIconSize(QtCore.QSize(18, 18))
            button.setFixedSize(32, 32)

        # self.path_label.setReadOnly(True)

        l = QtGui.QVBoxLayout(self)
        l.setStretch(0, 0)
        l.setContentsMargins(0, 0, 0, 0)
        l.setSpacing(0)
        hl = QtGui.QHBoxLayout()
        hl.addWidget(self.back_button)
        hl.addWidget(self.refresh_button)
        hl.addWidget(self.path_label)
        hl.setStretch(0, 0)
        l.addLayout(hl)
        l.addWidget(self.file_view)

        self.setPathLabel(self.file_view.path)
        self.back_button.clicked.connect(self.onBackClicked)
        self.refresh_button.clicked.connect(self.onRefreshClicked)
        self.file_view.pathChanged.connect(self.setPathLabel)
        self.path_label.textChanged.connect(self.pathlabelChanged)

    def pathlabelChanged(self):
        path = self.path_label.text()
        self.file_view.refresh(path=path)

    def onBackClicked(self):
        path = self.file_view.path
        path = os.path.dirname(str(path))
        self.file_view.refresh(path=path)

    def onRefreshClicked(self):
        self.file_view.refresh()

    def setPathLabel(self, path):
        self.path_label.setText(path)

    def getSelectedFilePaths(self):
        return self.file_view.getSelectedFilePaths()


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
        return os.path.join(*self.file_view.getStagesAndDatasets())

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


class MultipleFileExplorer(QtGui.QTabWidget):
    """
    Class for multiple location file explorer, capability to add GlobusFileView Tabs
    """

    sigLoginSuccess = QtCore.Signal(bool)
    sigLoginRequest = QtCore.Signal(QtCore.Signal, bool)
    sigProgJob = QtCore.Signal(str, object, list, dict, object)
    sigPulsJob = QtCore.Signal(str, object, list, dict, object)
    sigSFTPJob = QtCore.Signal(str, object, list, dict, object)
    sigOpen = QtCore.Signal(list)
    sigFolderOpen = QtCore.Signal(list)
    sigPreview = QtCore.Signal(object)

    def __init__(self, parent=None):
        super(MultipleFileExplorer, self).__init__(parent)
        self.explorers = OrderedDict()

        self.tab = TabBarPlus()
        self.setTabBar(self.tab)
        self.setTabsClosable(True)

        self.explorers['Local'] = FileExplorer(LocalFileView(self), self)
        self.addFileExplorer('Local', self.explorers['Local'], closable=False)

        self.jobtab = JobTable(self)
        # Do not understand why I need to add it and remove it so that its not added as a seperate widget
        self.addTab(self.jobtab, 'Jobs')
        self.removeTab(1)

        self.sigProgJob.connect(self.jobtab.addProgJob)
        self.sigPulsJob.connect(self.jobtab.addPulseJob)
        self.sigSFTPJob.connect(self.jobtab.addSFTPJob)

        self.tab.plusClicked.connect(self.onPlusClicked)
        self.tabCloseRequested.connect(self.removeTab)

        self.newtabmenu = QtGui.QMenu(None)
        addspot = QtGui.QAction('SPOT', self.newtabmenu)
        addcori = QtGui.QAction('Cori', self.newtabmenu)
        addedison = QtGui.QAction('Edison', self.newtabmenu)
        addbragg = QtGui.QAction('Bragg', self.newtabmenu)
        addsftp = QtGui.QAction('SFTP Connection', self.newtabmenu)
        showjobtab = QtGui.QAction('Jobs', self.newtabmenu)
        self.standard_actions = OrderedDict({'SPOT': addspot, 'Cori': addcori, 'Edison': addedison,
                                             'Bragg': addbragg, 'SFTP': addsftp})
        self.newtabmenu.addActions(self.standard_actions.values())
        self.newtabmenu.addAction(showjobtab)
        addspot.triggered.connect(self.addSPOTTab)
        addedison.triggered.connect(lambda: self.addHPCTab('Edison'))
        addcori.triggered.connect(lambda: self.addHPCTab('Cori'))
        addbragg.triggered.connect(lambda: self.addHPCTab('Bragg'))
        addsftp.triggered.connect(self.addSFTPTab)
        showjobtab.triggered.connect(lambda: self.addTab(self.jobtab, 'Jobs'))

    def enableActions(self):
        for name, action in self.standard_actions.iteritems():
            if name in self.explorers.keys():
                action.setEnabled(False)
            else:
                action.setEnabled(True)

    def removeTab(self, p_int):
        if self.tabText(p_int) != 'Jobs':
            name = self.explorers.keys()[p_int]
            explorer = self.explorers.pop(name)
            cmanager.logout(explorer.file_view.client)
            self.widget(p_int).deleteLater()
            self.enableActions()
        super(MultipleFileExplorer, self).removeTab(p_int)

    def removeTabs(self):
        for i in xrange(1, self.count()):
            self.removeTab(1)

    def onPlusClicked(self):
        self.newtabmenu.popup(QtGui.QCursor.pos())

    def addFileExplorer(self, name, file_explorer, closable=True):
        self.explorers[name] = file_explorer
        file_explorer.file_view.sigItemPreview.connect(self.itemSelected)
        self.wireExplorerSignals(file_explorer)
        idx = len(self.explorers) - 1
        tab = self.insertTab(idx, file_explorer, name)
        if closable is False:
            try:
                self.tabBar().tabButton(tab, QtGui.QTabBar.RightSide).resize(0, 0)
                self.tabBar().tabButton(tab, QtGui.QTabBar.RightSide).hide()
            except AttributeError:
                self.tabBar().tabButton(tab, QtGui.QTabBar.LeftSide).resize(0, 0)
                self.tabBar().tabButton(tab, QtGui.QTabBar.LeftSide).hide()
        self.setCurrentWidget(file_explorer)

    def wireExplorerSignals(self, explorer):
        explorer.file_view.sigOpen.connect(self.handleOpenActions)
        explorer.file_view.sigOpenFolder.connect(self.handleOpenFolderActions)
        try:
            explorer.file_view.sigDownload.connect(self.handleDownloadActions)
        except AttributeError:
            pass
        try:
            explorer.file_view.sigDelete.connect(self.handleDeleteActions)
        except AttributeError:
            pass
        try:
            explorer.file_view.sigTransfer.connect(self.handleTransferActions)
        except AttributeError:
            pass

    def itemSelected(self, item):
        msg.clearMessage() #
        self.sigPreview.emit(item)

    def addHPCTab(self, system):
        # # NERSC tabs based on NEWT API
        # add_nersc_explorer = lambda client: self.addFileExplorer(system.capitalize(),
        # FileExplorer(NERSCFileView(client, system, self)))
        # login_callback = lambda client: self.loginSuccess(client, add_explorer=add_nersc_explorer)
        # self.sigLoginRequest.emit(partial(cmanager.login, login_callback, cmanager.spot_client.login), False)
        # add_sftp_explorer = lambda client: self.addFileExplorer(client.host.split('.')[0],
        #                                                         FileExplorer(SFTPTreeWidget(client, self)))

        # NERSC tabs based on SFTP
        add_sftp_explorer = lambda client: self.addFileExplorer(system,
                                                                FileExplorer(SFTPFileView(client, self)))
        add_sftp_callback = lambda client: self.loginSuccess(client,
                                                             add_explorer=add_sftp_explorer)
        login_callback = lambda client: cmanager.add_sftp_client(system,
                                                                 client,
                                                                 add_sftp_callback)
        sftp_client = partial(cmanager.sftp_client, cmanager.HPC_SYSTEM_ADDRESSES[system])
        self.sigLoginRequest.emit(partial(cmanager.login, login_callback, sftp_client), False)

    def addSPOTTab(self):
        add_spot_explorer = lambda client: self.addFileExplorer('SPOT',
                                                                SpotDatasetExplorer(client, self))
        login_callback = lambda client: self.loginSuccess(client,
                                                          add_explorer=add_spot_explorer)
        self.sigLoginRequest.emit(partial(cmanager.login, login_callback, cmanager.spot_client.login), False)

    #TODO add globus tranfer capabilities to SFTP connected machines if they are globus endpoints
    # This is being replaced by SFTP tabs
    # def addGlobusTab(self, endpoint):
    # add_globus_explorer = lambda client: self.addFileExplorer(endpoint.split('#')[-1],
    #                                                              FileExplorer(GlobusFileView(client, client, self)))
    #     add_endpoint_callback = lambda client: self.loginSuccess(client,
    #                                                              add_explorer=add_globus_explorer)
    #     login_callback = lambda client: cmanager.add_globus_client(endpoint.split('#')[-1],
    #                                                                client,
    #                                                                add_endpoint_callback)
    #     globus_client = cmanager.globus_client()
    #     self.sigLoginRequest.emit(partial(cmanager.login, login_callback, globus_client.login), False)

    def addSFTPTab(self):
        add_sftp_explorer = lambda client: self.addFileExplorer(client.host.split('.')[0],
                                                                FileExplorer(SFTPFileView(client, self)))
        add_sftp_callback = lambda client: self.loginSuccess(client,
                                                             add_explorer=add_sftp_explorer)
        login_callback = lambda client: cmanager.add_sftp_client(client.host,
                                                                 client,
                                                                 add_sftp_callback)
        sftp_client = cmanager.sftp_client
        self.sigLoginRequest.emit(partial(cmanager.login, login_callback, sftp_client), True)

    def loginSuccess(self, client, add_explorer=None):
        if not client:
            self.sigLoginSuccess.emit(False)
        else:
            add_explorer(client)
            self.enableActions()
            self.sigLoginSuccess.emit(True)

    def getSelectedFilePaths(self):
        return self.currentWidget().getSelectedFilePaths()

    def getCurrentPath(self):
        return self.currentWidget().path

    def getPath(self, tab_name):
        return self.explorers[tab_name].path

    def handleOpenActions(self, paths):
        if len(paths) > 0:
            self.sigOpen.emit(paths)

    def handleOpenFolderActions(self, paths):
        if len(paths) > 0:
            self.sigFolderOpen.emit(paths)

    def handleDeleteActions(self, paths):
        r = QtGui.QMessageBox.warning(self, 'Delete file',
                                      'Are you sure you want to delete\n{}?'.format(',\n'.join(paths)),
                                      QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)
        if r == QtGui.QMessageBox.Yes:
            self.currentWidget().file_view.deleteSelection()

    def handleUploadAction(self, desc, method, args, kwargs):
        self.sigProgJob.emit(desc, method, args, kwargs)
        self.addTab(self.jobtab, 'Jobs')

    def handleDownloadActions(self, name, desc, method, args, kwargs, fslot):
        if 'save_path' not in kwargs and 'localpath' not in kwargs:
            fileDialog = QtGui.QFileDialog(self, 'Save as', os.path.expanduser('~'))
            fileDialog.setAcceptMode(QtGui.QFileDialog.AcceptSave)
            fileDialog.selectFile(name)
            if fileDialog.exec_():
                save_path = str(fileDialog.selectedFiles()[0])
                if isinstance(self.currentWidget().file_view, SFTPFileView):
                    kwargs['localpath'] = save_path
                else:
                    kwargs['save_path'] = save_path
        if isinstance(self.currentWidget().file_view, SFTPFileView):
            self.sigSFTPJob.emit(desc, method, args, kwargs, fslot)
        else:
            self.sigProgJob.emit(desc, method, args, kwargs, fslot)
        self.addTab(self.jobtab, 'Jobs')

    def handleTransferActions(self, paths, desc, method, args, kwargs):
        #TODO Need to implement this
        if isinstance(self.currentWidget().file_view, SpotDatasetView):
            self.sigPulsJob.emit(desc, method, args, kwargs)
        else:
            self.sigProgJob.emit(desc, method, args, kwargs)
        self.addTab(self.jobtab, 'Jobs')


class TabBarPlus(QtGui.QTabBar):
    """
    Tab bar that has a plus button floating to the right of the tabs.
    """

    plusClicked = QtCore.Signal()

    def __init__(self, parent=None):
        super(TabBarPlus, self).__init__(parent)

        self.plus_button = QtGui.QPushButton(" + ")
        self.plus_button.setToolTip('Add browser location')
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
        return QtCore.QSize(width + 32, height)

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


class JobTable(QtGui.QTableWidget):
    """
    Class with table of download, upload and transfer jobs entries.
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

        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.menu = QtGui.QMenu(self)
        cancel = QtGui.QAction('Cancel', self)
        cancel.triggered.connect(self.cancelActionTriggered)
        remove = QtGui.QAction('Remove', self)
        remove.triggered.connect(self.removeActionTriggered)
        self.menu.addActions([cancel, remove])
        self.customContextMenuRequested.connect(self.menuRequested)

    def menuRequested(self, pos):
        self.menu.exec_(self.mapToGlobal(pos))

    def cancelActionTriggered(self):
        self.jobs[self.currentRow()].cancel()

    def removeActionTriggered(self):
        self.removeJob(self.jobs[self.currentRow()])

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
        job_entry = self.addJob(job_desc)
        bg_generator = threads.iterator(callback_slot=job_entry.progress,
                                        finished_slot=finish_slot,
                                        interrupt_signal=job_entry.sigCancel)(generator)
        bg_generator(*args, **kwargs)

    def addSFTPJob(self, job_desc, method, args, kwargs, finish_slot=None):
        job_entry = self.addJob(job_desc)
        kwargs['callback'] = job_entry.progressRaw
        bg_method = threads.method(finished_slot=finish_slot)(method)
        bg_method(*args, **kwargs)

    @QtCore.Slot(str, object, list, dict)
    def addPulseJob(self, job_type, job_desc, method, args, kwargs):
        job_entry = self.addJob(job_type, job_desc)
        job_entry.sigRemove.connect(self.removeJob)
        job_entry.pulseStart()
        bg_method = threads.method(finished_slot=job_entry.pulseStop, interrupt_signal=job_entry.sigCancel)(method)
        bg_method(*args, **kwargs)


    def removeJob(self, jobentry):
        jobentry.cancel()
        idx = self.jobs.index(jobentry)
        del self.jobs[idx]
        self.removeRow(idx)
        jobentry.deleteLater()


class JobEntry(QtGui.QWidget):
    """
    Class for job entries (downloads/uploads/transfers) in Job Table. Each job entry has a description and a progress
    bar to show fractional prograss or pulsing.
    """

    sigCancel = QtCore.Signal()
    sigRemove = QtCore.Signal(object)

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

    def cancel(self):
        self.sigCancel.emit()

    def progress(self, i):
        i = int(i*100)
        self.progressbar.setValue(i)

    def progressRaw(self, transfered, total):
        self.progress(transfered/total)

    def pulseStart(self):
        self.progressbar.setRange(0, 0)

    def pulseStop(self):
        self.progressbar.setRange(0, 100)
        self.progress(1)


class LazyTreeItem(QtGui.QTreeWidgetItem):
    """
    Base class for a lazy tree item that does not get its children until asked to
    """

    def __init__(self, name, icon=None, parent=None):
        super(LazyTreeItem, self).__init__([name], parent=parent)
        self.parentItem = parent
        if icon is not None:
            self.setIcon(0, icon)
        self.setChildIndicatorPolicy(QtGui.QTreeWidgetItem.ShowIndicator)

    def setExpanded(self, expand):
        if expand and self.childCount() == 0:
            self.getChildren()
        super(LazyTreeItem, self).setExpanded(expand)

    def getChildren(self):
        # Override this function to get children
        raise NotImplementedError('getChildren method not implemented. This method must be implemented.')


# TODO walktree in a different thread!
class SFTPDirTreeItem(LazyTreeItem, QtCore.QObject):
    """
    SFTP folder tree item that will not retrieve its contents until it is expanded
    """
    sigAddChildFile = QtCore.Signal(str)
    sigAddChildDir = QtCore.Signal(str)

    def __init__(self, name, client, path, icon=None, parent=None):
        # super(SFTPDirTreeItem, self).__init__(name, icon, parent)
        LazyTreeItem.__init__(self, name, icon, parent)
        QtCore.QObject.__init__(self, parent=parent)
        self.client = client
        self.path = path
        self.sigAddChildFile.connect(self.addChildFile)
        self.sigAddChildDir.connect(self.addChildDir)

    @threads.method()
    def getChildren(self):
        self.client.walktree(self.path, self.sigAddChildFile.emit, self.sigAddChildDir.emit,
                             self.handleUnknown, recurse=False)

    def addChildFile(self, path):
        name = os.path.split(path)[-1]
        icon = QtGui.QFileIconProvider().icon(QtGui.QFileIconProvider.File)
        item = SFTPFileTreeItem(name, path, icon=icon, parent=self)
        item.setIcon(0, icon)
        self.addChild(item)


    def addChildDir(self, path):
        name = os.path.split(path)[-1]
        icon = QtGui.QFileIconProvider().icon(QtGui.QFileIconProvider.Folder)
        item = SFTPDirTreeItem(name, self.client, path, icon, self)
        item.setIcon(0, icon)
        self.addChild(item)

    def handleUnknown(self, path):
        msg.logMessage('Unknown object found: {0}'.format(path),msg.WARNING)


class SFTPFileTreeItem(QtGui.QTreeWidgetItem):
    """
    SFTP File tree item that saves its path
    """

    def __init__(self, name, path, icon=None, parent=None):
        super(SFTPFileTreeItem, self).__init__([name], parent=parent)
        self.parentItem = parent
        if icon is not None:
            self.setIcon(0, icon)
        self.path = path


if __name__ == '__main__':
    import client
    import getpass
    import sys

    passw = getpass.getpass()
    client = client.sftp.SFTPClient('bl832viz2.dhcp.lbl.gov', username='lbluque', password=passw)
    app = QtGui.QApplication(sys.argv)
    w = SFTPFileView(client)
    w.setWindowTitle("Test this thing")
    w.show()
    sys.exit(app.exec_())