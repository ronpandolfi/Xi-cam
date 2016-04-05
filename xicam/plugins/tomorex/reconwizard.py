# -*- coding: utf-8 -*-
"""
@author: lbluque
"""
#TODO Metadata from SPOT and NERSC
#TODO Job parameters for NERSC
#TODO Run jobs on NERSC

import os
from copy import deepcopy

import psutil
from PySide import QtCore, QtGui
from PySide.QtUiTools import QUiLoader
from numpy import round

#from spew import jobs
import tomopynames as tp
import treader
from scriptmaker import write_tomopy_script


RECON_PROGRAMS = ['tomopy', 'MBIR']
OUTPUT_FORMATS = {'TIFF': '.tiff', 'BL832 HDF5': '.h5', 'Data-exchange': '.h5', 'NumPy bin': '.npy'}
JOB_DEFAULT_PARAMS = {'ncore': None, 'nchunk': 1}
# nchunk here does not refer to the kwarg in tomopy functions
# but to the number of chunks in which a dataset will be processed


class Wizard(QtGui.QWidget):
    """
    Class that handles input files, reconstruction program, and run location
    selection.
    """

    sigExternalJob = QtCore.Signal(str, str, object, list, dict)

    def __init__(self, fileexplorer=None, parent=None):
        super(Wizard, self).__init__(parent)
        self.ui = QUiLoader().load('xicam/plugins/tomorex/gui/recon_setup.ui', self)
        self.fileexplorer = fileexplorer

        path = fileexplorer.getSelectedFilePath()
        self.input_path = path if path is not None else ''
        self.recon_programs = {}
        for program in RECON_PROGRAMS:
            self.recon_programs[program] = False
        self.tomopy_options = deepcopy(tp.DEFAULT_OPTIONS)
        self.tomopy_params = deepcopy(tp.DEFAULT_PARAMS)
        self.recon_indices = ([0, 0, 1], [0, 0, 1]) # ([sino], [proj])
        self.job_params = deepcopy(JOB_DEFAULT_PARAMS)
        self.dset_metadata = {}

        self.ui.input_lineEdit.setText(self.input_path)
        self.ui.output_comboBox.addItems(OUTPUT_FORMATS.keys())
        file_locations = fileexplorer.explorers.keys()
        run_locations = file_locations[:2]
        self.ui.input_comboBox.addItems(file_locations)
        self.ui.run_comboBox.addItems(run_locations)

        self.checkAvailableReconPrograms()
        self.addRunPrograms()

        index = self.ui.input_comboBox.findText('Local', QtCore.Qt.MatchExactly)
        self.ui.input_comboBox.setCurrentIndex(index)

        self.adjustRunLocation()
        self.output_path = self.defOutputPath(self.input_path)
        self.ui.output_lineEdit.setText(self.output_path)

        self.ui.run_comboBox.currentIndexChanged.connect(self.addRunPrograms)
        self.ui.run_comboBox.currentIndexChanged.connect(self.adjustOutputPath)
        self.ui.input_comboBox.currentIndexChanged.connect(self.setSelectedFile)
        self.ui.input_comboBox.currentIndexChanged.connect(self.adjustRunLocation)
        self.ui.input_lineEdit.textChanged.connect(self.adjustOutputPath)
        self.ui.input_button.clicked.connect(self.selectInputFile)
        self.ui.output_button.clicked.connect(self.selectOutputFile)
        self.ui.recon_button.clicked.connect(self.getReconOptions)
        self.ui.job_button.clicked.connect(self.getJobParameters)
        self.ui.output_comboBox.currentIndexChanged.connect(self.adjustFileExtension)
        self.ui.run_button.clicked.connect(self.onRunClicked)
        self.ui.cancel_button.clicked.connect(self.ui.reject)

        if self.ui.input_lineEdit.text() != '':
            self.dset_metadata = self.getFileMetadata(str(self.ui.input_lineEdit.text()),
                                                      str(self.ui.input_comboBox.currentText()))
            self.setDefaultsFromMetadata()

    def exec_(self):
        self.ui.exec_()

    def onRunClicked(self):
        if '.h5' in self.ui.input_lineEdit.text():
            if self.ui.output_lineEdit != '':
                script_path = write_tomopy_script(str(self.ui.input_lineEdit.text()),
                                                  str(self.ui.output_lineEdit.text()),
                                                  self.dset_metadata,
                                                  self.tomopy_options,
                                                  self.tomopy_params,
                                                  self.recon_indices,
                                                  self.job_params,
                                                  str(self.ui.output_comboBox.currentText()))

                if self.ui.run_comboBox.currentText() == 'Local':
                    self.sigExternalJob.emit('Reconstruction',
                                             'Reconstructing {}'.format(self.ui.input_lineEdit.text()),[])
                                             #jobs.local_script_generator, ['python', script_path], {})
                    # proc = jobs.run_local_script('python', script_path)
                    # self.sigExternalJob.emit('Reconstruction',
                    #                          'Reconstructing {}'.format(self.ui.input_lineEdit.text()),
                    #                          jobs.script_status_generator, [proc.poll, proc.communicate], {})
                elif self.ui.run_comboBox.currentText() == 'NERSC':
                    # r = self.client.spot.upload_file(script_path, str(self.client.scratch_dir), system='cori')
                    # r = self.client.spot.execute_command('python ' + script_path, 'cori', bin_path='usr/bin')
                    # r = self.client.spot.execute_command('python ' + script_path, 'cori', bin_path='usr/bin')
                    r = 'naaps'
                    print r
                self.ui.accept()
            else:
                QtGui.QMessageBox.information(self, 'No output selected', 'Please select an output name')
                return
        else:
            QtGui.QMessageBox.information(self, 'Not an H5 file', 'Please select an H5 file as input')

    def checkImport(self, module):
        try:
            exec('import ' + module)
            return True
        except ImportError:
            return False

    def checkAvailableReconPrograms(self):
        for program in self.recon_programs:
            if program == 'MBIR':
                if self.fileexplorer.nersc_login is True:
                    self.recon_programs[program] = True
                else:
                    self.recon_programs[program] = False
            elif self.checkImport(program):
                self.recon_programs[program] = True
            else:
                QtGui.QMessageBox.warning(self, '{0} import error'.format(program), '{0} could not be imported.\n'
                                          'Make sure {0} is installed correctly and can be imported by '
                                          'Python. '.format(program))

    def addRunPrograms(self):
        self.ui.recon_comboBox.clear()
        if self.ui.run_comboBox.currentText() == 'Local' and self.recon_programs['tomopy']:
            self.ui.recon_comboBox.addItem('tomopy')
        else:
            self.ui.recon_comboBox.addItems(self.recon_programs.keys())

    def adjustOutputPath(self):
        if self.ui.input_lineEdit.text() == '':
            self.ui.output_lineEdit.clear()
            return

        fname = os.path.split(str(self.ui.input_lineEdit.text()))[-1]

        path = self.fileexplorer.explorers[str(self.ui.run_comboBox.currentText())]
        if self.ui.run_comboBox.currentText() == 'Local':
            if self.ui.input_comboBox.currentText() == 'Local':
                path = str(self.ui.input_lineEdit.text())
            else:
                path = os.path.join(os.path.expanduser('~'), fname)
        else:
            path = self.fileexplorer.explorers[''] + '/' + fname

        if self.ui.input_comboBox.currentText() == 'SPOT':
            path = fname

        path = self.defOutputPath(path)
        self.ui.output_lineEdit.setText(path)

    def adjustRunLocation(self):
        if self.ui.input_comboBox.currentText() == 'Local':
            index = 0
        else:
            index = 1
        self.ui.run_comboBox.setCurrentIndex(index)

    def adjustFileExtension(self):
        file_type = self.ui.output_comboBox.currentText()
        path = str(self.ui.output_lineEdit.text()).os.path.split('.')[0]
        if file_type in ('BL832 HDF5', 'Data-exchange'):
            path += '.h5'
        elif file_type == 'TIFF':
            path += '.tiff'
        elif file_type == 'NumPy bin':
            path += '.npy'

        self.ui.output_lineEdit.setText(path)

    def adjustComboBoxExtension(self):
        extension = str(self.ui.output_lineEdit.text()).os.path.split('.')[-1]

        if extension in '.h5':
            file_type = 'BL832 HDF5'
        elif extension in '.npy':
            file_type = 'NumPy bin'
        elif extension in ('.tiff', '.TIFF', '.tif', '.TIF'):
            file_type = 'TIFF'
        else:
            return

        index = self.ui.run_comboBox.findText(file_type, QtCore.Qt.MatchExactly)
        self.ui.run_comboBox.setCurrentIndex(index)

    def setSelectedFile(self):
        self.ui.input_lineEdit.clear()
        path = self.fileexplorer.getSelectedFilePath()
        self.ui.input_lineEdit.setText(path)

    def selectInputFile(self):
        file_selected = False
        location = str(self.ui.input_comboBox.currentText())
        if location == 'Local':
            fileDialog = QtGui.QFileDialog(self, 'Select h5 file', os.path.expanduser('~'))
            fileDialog.setNameFilters(['*.h5'])  # , '*.tiff', '*.tif'])
            if fileDialog.exec_():
                path = str(fileDialog.selectedFiles()[0])
                file_selected = True
        else:
            items = self.fileexplorer.explorers[location].getRawDatasetList()
            item, ok = QtGui.QInputDialog.getItem(self, 'Select h5 file', 'Dataset:', items, 0, False)
            if ok:
                file_selected = True
                if location == 'SPOT':
                    path = item
                else:
                    path = self.fileexplorer.getPath(location) + '/' + item

        if file_selected is True:
            self.ui.input_lineEdit.setText(path)
            self.adjustOutputPath()
            self.dset_metadata = self.getFileMetadata(str(self.ui.input_lineEdit.text()),
                                                      str(self.ui.input_comboBox.currentText()))
            self.setDefaultsFromMetadata()

    def setDefaultsFromMetadata(self):
        if len(self.dset_metadata) > 0:
            self.recon_indices[0][1] = int(self.dset_metadata['nslices'])
            self.recon_indices[1][1] = int(self.dset_metadata['nangles'])
            self.tomopy_params['manual']['value'] = float(self.dset_metadata['nrays'])/2.0
            self.tomopy_params['Paganin']['pixel_size'] = float(self.dset_metadata['pxsize'])/10.0
            self.tomopy_params['Paganin']['dist'] = float(self.dset_metadata['Camera_Z_Support'])
            if self.dset_metadata['senergy'] == 'Inf':
                self.tomopy_params['Paganin']['energy'] = 40.0
            else:
                self.tomopy_params['Paganin']['energy'] = float(self.dset_metadata['senergy'])/1000.0

    def defOutputPath(self, path):
        if self.ui.input_lineEdit.text() == '':
            return ''

        loc = self.ui.input_comboBox.currentText()
        if loc == 'SPOT':
            stage, name = self.fileexplorer.getPath(str(loc)).split('/')
            name += 'RECON_' + name
            if self.ui.run_comboBox.currentText() == 'Local':
                path = os.path.expanduser('~')
            else:
                path = self.fileexplorer.explorers[loc].path
        else:
            name = 'RECON_' + os.path.split(path)[-1]
            path = os.path.split(path)[0]

        if self.ui.run_comboBox.currentText() == 'Local':
            path = os.path.join(path, name)
        else:
            path = path + '/' + name

        fmt = str(self.ui.output_comboBox.currentText())
        ext = OUTPUT_FORMATS[fmt]
        if fmt == 'TIFF':
            path = os.path.join(path.split('.')[0], os.path.split(path)[1].split('.')[0] + ext)
        else:
            path = path.os.path.split('.')[0] + ext

        return path

    def selectOutputFile(self):
        if self.ui.run_comboBox.currentText() == 'Local':
            path = os.path.expanduser('~') if self.ui.input_lineEdit.text() == '' \
                   else os.path.split(str(self.ui.output_lineEdit.text()))[0]
            fileDialog = QtGui.QFileDialog(self, 'Save output as', path)
            fileDialog.setNameFilters(['*.h5', '*.tiff', '*.tif'])
            fileDialog.setAcceptMode(QtGui.QFileDialog.AcceptSave)
            fileDialog.selectFile(self.output_lineEdit.text())

            if fileDialog.exec_():
                outname = str(fileDialog.selectedFiles()[0])
                self.ui.output_lineEdit.setText(outname)
        else:
            path = self.defOutputPath(str(self.ui.input_lineEdit.text()))
            item, ok = QtGui.QInputDialog.getText(self, 'Output path and name',
                                                  'Provide the output path and'
                                                  ' filename', text=path)
            if ok:
                self.ui.output_lineEdit.setText(item)

    def getReconOptions(self):
        if self.ui.input_lineEdit.text() == '':
            QtGui.QMessageBox.information(self, 'No input dataset selected',
                                          'Please select an input dataset.')
        elif self.ui.recon_comboBox.currentText() == 'tomopy':
            reconOptions = TomopyReconOptionsWindow(self.tomopy_options, self.dset_metadata, self.recon_indices, self)
            if reconOptions.exec_():
                self.readReconOptions(reconOptions)

    def readReconOptions(self, reconOptions):
        for option in tp.OPTIONS:
            if option in ('downsample', 'upsample', 'pad', 'mask'):
                box_name = option + '_checkBox'
                box = reconOptions.findChild(QtGui.QCheckBox, box_name)
                self.tomopy_options[option] = box.isChecked()
            else:
                box_name = option + '_comboBox'
                box = reconOptions.findChild(QtGui.QComboBox, box_name)
                self.tomopy_options[option] = str(box.currentText())
        sino_indices = (reconOptions.ui.slice_start_spinBox.value(),
                        reconOptions.ui.slice_end_spinBox.value(),
                        reconOptions.ui.slice_step_spinBox.value())
        proj_indices = (reconOptions.ui.proj_start_spinBox.value(),
                        reconOptions.ui.proj_end_spinBox.value(),
                        reconOptions.ui.proj_step_spinBox.value())
        self.recon_indices = (sino_indices, proj_indices)

        # Adjust default center of rotation in reconstruction
        if self.tomopy_options['pad'] is True:
            self.tomopy_params[self.tomopy_options['recon']]['center'] = 'cor + npad'
        else:
            self.tomopy_params[self.tomopy_options['recon']]['center'] = 'cor'

    def getReconParameters(self):
        if self.ui.recon_comboBox.currentText() == 'tomopy':
            reconParams = TomopyParametersWindow(deepcopy(self.tomopy_options),
                                                 deepcopy(self.tomopy_params),
                                                 self)
            if reconParams.exec_():
                self.readReconParameters(reconParams)

    def readReconParameters(self, reconParams):
        for i, (func, params) in enumerate(reconParams.params.iteritems()):
            row = None
            items = reconParams.table.findItems(func, QtCore.Qt.MatchExactly)
            for item in items:
                if item.column() == 1:
                    row = item.row()
            if row is None:
                continue

            for j, (param, value) in enumerate(params.iteritems()):
                if param == 'algorithm' or param == 'center':
                    continue
                row += 1
                param_item = reconParams.table.item(row, 0)
                param = str(param_item.text())
                value_item = reconParams.table.item(row, 1)
                if value_item is None:
                    value_item = reconParams.table.cellWidget(row, 1)
                    if isinstance(value_item, QtGui.QComboBox):
                        value = str(value_item.currentText())
                    elif isinstance(value_item, QtGui.QCheckBox):
                        value = value_item.isChecked()
                    elif isinstance(value_item, QtGui.QAbstractSpinBox):
                        value = value_item.value()
                else:
                    value = str(value_item.text())
                self.tomopy_params[func][param] = value

    def getFileMetadata(self, file_name, location):
        if location == 'Local':
            message = None
            try:
                data = treader.read_als_832h5_metadata(file_name)
            except IOError:
                message = 'Unable to open {}'.format(file_name)
            except:
                message = 'The format of the dataset selected is not recognized'

            if message is not None:
                QtGui.QMessageBox.warning(self, 'Input error', message)
                return

        elif location == 'SPOT':
            dataset, stage = self.fileexplorer['SPOT'].getPath().split('/')
            # data =
        else:
            # get metadata from nersc
            print 'Not implemented'
            #data = self.parent.worker.spot.execute_cmd()

        return data

    def getJobParameters(self):
        if self.ui.run_comboBox.currentText() == 'Local':
            jobOptions = LocalJobParametersWindow(self.job_params, self)
            if jobOptions.exec_():
                self.job_params['ncore'] = jobOptions.ui.core_spinBox.value()
                self.job_params['nchunk'] = jobOptions.ui.chunk_spinBox.value()
                if self.job_params['ncore'] == psutil.cpu_count():
                    self.job_params['ncore'] = None
                if self.job_params['nchunk'] == 1:
                    self.job_params['nchunk'] = None
        else:
            pass


class TomopyReconOptionsWindow(QtGui.QWidget):
    """
    Class that handles input of tomopy reconstruction options
    """

    def __init__(self, options, dset_metadata, recon_indices, parent=None):
        super(TomopyReconOptionsWindow, self).__init__(parent)
        self.parent = parent
        self.options = options
        self.dset_medatata = dset_metadata
        self.ui = QUiLoader().load('xicam/plugins/tomorex/gui/tomopy_recon_options.ui', self)
        self.ui.next_button.clicked.connect(self.getReconParameters)
        self.ui.save_button.clicked.connect(self.saveScript)
        self.fillComboBoxes()
        self.setDefaultOptions(recon_indices)

    def exec_(self):
        self.ui.exec_()

    def show(self):
        self.ui.show()
        self.ui.raise_()

    def fillComboBoxes(self):
        for option in tp.OPTIONS:
            if option in ('downsample', 'upsample', 'pad', 'mask'):
                continue
            box_name = option + '_comboBox'
            self.ui.findChild(QtGui.QComboBox, box_name).addItems(tp.OPTIONS[option])

    def setDefaultOptions(self, recon_indices):
        max_slice = int(self.dset_medatata['nslices'])
        max_proj = int(self.dset_medatata['nangles'])
        self.ui.slice_start_spinBox.setRange(0, max_slice)
        self.ui.slice_end_spinBox.setRange(0, max_slice)
        self.ui.slice_step_spinBox.setRange(0, max_slice)
        self.ui.proj_start_spinBox.setRange(0, max_proj)
        self.ui.proj_end_spinBox.setRange(0, max_proj)
        self.ui.proj_step_spinBox.setRange(0, max_proj)
        self.ui.slice_start_spinBox.setValue(recon_indices[0][0])
        self.ui.slice_end_spinBox.setValue(recon_indices[0][1])
        self.ui.slice_step_spinBox.setValue(recon_indices[0][2])
        self.ui.proj_start_spinBox.setValue(recon_indices[1][0])
        self.ui.proj_end_spinBox.setValue(recon_indices[1][1])
        self.ui.proj_step_spinBox.setValue(recon_indices[1][2])

        for option in self.options:
            box_name = option
            if option in ('downsample', 'upsample', 'pad', 'mask'):
                box_name += '_checkBox'
                checkBox = self.ui.findChild(QtGui.QCheckBox, box_name)
                checkBox.setChecked(self.options[option])
            else:
                box_name += '_comboBox'
                comboBox = self.ui.findChild(QtGui.QComboBox, box_name)
                index = comboBox.findText(self.options[option],
                                          QtCore.Qt.MatchExactly)
                comboBox.setCurrentIndex(index)

    def getReconParameters(self):
        self.parent.readReconOptions(self)
        self.parent.getReconParameters()

    def saveScript(self):
        self.parent.readReconOptions(self)
        save_name = QtGui.QFileDialog.getSaveFileName(self, 'Save TomoPy Script',
                                                      os.path.expanduser('~'),
                                                      '*.py')
        if '.py' not in save_name:
            save_name += '.py'

        write_tomopy_script(str(self.parent.ui.input_lineEdit.text()),
                            str(self.parent.ui.output_lineEdit.text()),
                            self.parent.dset_metadata,
                            self.parent.tomopy_options,
                            self.parent.tomopy_params,
                            self.parent.recon_indices,
                            self.parent.job_params,
                            str(self.parent.ui.output_comboBox.currentText()),
                            save_as=str(save_name))


class TomopyParametersWindow(QtGui.QDialog):
    """
    Class that handles input of tomopy reconstruction parameters
    """
    def __init__(self,options, params, parent=None):
        super(TomopyParametersWindow, self).__init__(parent)
        self.options = options
        self.params = self.createParamsDict(params)
        self.param_dict = self.createParamsDict(deepcopy(tp.PARAMS))
        self.setupUi()
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.resetButton.clicked.connect(self.restoreDefaults)

    def setupUi(self):
        self.setWindowTitle('TomoPy Reconstruction Parameters')
        vlayout = QtGui.QVBoxLayout(self)
        self.table = QtGui.QTableWidget(0, 3)
        vlayout.addWidget(self.table)
        self.buttonBox = QtGui.QDialogButtonBox(self)
        self.okButton = self.buttonBox.addButton(QtGui.QDialogButtonBox.Ok)
        self.cancelButton = self.buttonBox.addButton(QtGui.QDialogButtonBox.Cancel)
        self.resetButton = self.buttonBox.addButton(QtGui.QDialogButtonBox.RestoreDefaults)
        vlayout.addWidget(self.buttonBox)
        self.buildTable(self.table, self.param_dict)
        self.table.setFixedWidth(self.table.horizontalHeader().width()/2)
        self.adjustSize()
        #self.resizeWindow()

    def createParamsDict(self, indict):
        for key in self.options.keys():
            if type(self.options[key]) is bool and not self.options[key]:
                self.options.pop(key)
            elif self.options[key] == 'none':
                self.options.pop(key)
                if key == 'phase':
                    indict.pop('Paganin')
                # elif key == 'ring':
                #     indict.pop('')

        flat_options = [i for item in self.options.items() for i in item]
        [indict.pop(key) for key in indict.keys() if key not in flat_options]
        return indict

    def buildTable(self, table, dict):
        table.setHorizontalHeaderLabels(['Parameter', 'Value', 'Units'])
        table.verticalHeader().setVisible(False)
        row = 0
        for i, (func, params) in enumerate(dict.iteritems()):
            table.insertRow(row)
            item = QtGui.QTableWidgetItem(self.options.keys()[i])
            item.setFlags(QtCore.Qt.ItemIsEnabled)
            item.setTextAlignment(QtCore.Qt.AlignCenter)
            table.setItem(row, 0, item)
            table.item(row, 0).setBackground(QtGui.QColor(QtCore.Qt.darkGray))
            item = QtGui.QTableWidgetItem(func)
            item.setFlags(QtCore.Qt.ItemIsEnabled)
            item.setTextAlignment(QtCore.Qt.AlignCenter)
            table.setItem(row, 1, item)
            table.setSpan(row, 1, 1, 2)
            table.item(row, 1).setBackground(QtGui.QColor(QtCore.Qt.darkGray))
            row += 1
            for j, (param, values) in enumerate(params.iteritems()):
                table.insertRow(row)
                item = QtGui.QTableWidgetItem(param)
                item.setFlags(QtCore.Qt.ItemIsEnabled)
                table.setItem(row, 0, item)
                if isinstance(values[0], list):
                    comboBox = QtGui.QComboBox(self)
                    comboBox.addItems(values[0])
                    index = comboBox.findText(self.params[func][param])
                    comboBox.setCurrentIndex(index)
                    table.setCellWidget(row, 1, comboBox)
                elif values[0] is bool:
                    checkBox = QtGui.QCheckBox('True/False', self)
                    table.setCellWidget(row, 1, checkBox)
                    if self.params[func][param] is True:
                        checkBox.setCheckState(QtCore.Qt.Checked)
                elif values[0] is int or values[0] is float:
                    spinBox = QtGui.QSpinBox(self) if values[0] is int \
                              else QtGui.QDoubleSpinBox(self)
                    spinBox.setAlignment(QtCore.Qt.AlignCenter)
                    spinBox.setMaximum(999999999)
                    if self.params[func][param] is None:
                        spinBox.setSpecialValueText('None')
                    else:
                        spinBox.setValue(self.params[func][param])
                    table.setCellWidget(row, 1, spinBox)
                else:
                    item = QtGui.QTableWidgetItem(str(self.params[func][param]))
                    item.setTextAlignment(QtCore.Qt.AlignCenter)
                    table.setItem(row, 1, item)

                item = QtGui.QTableWidgetItem(values[1])
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                item.setFlags(QtCore.Qt.ItemIsEnabled)
                table.setItem(row, 2, item)
                row += 1

    def resizeWindow(self):
        s = self.sizeHint()
        s.setHeight(self.height())
        self.resize(s)

    def restoreDefaults(self):
        self.table.setRowCount(0)
        self.params = self.createParamsDict(deepcopy(tp.DEFAULT_PARAMS))
        self.param_dict = self.createParamsDict(deepcopy(tp.PARAMS))
        self.buildTable(self.table, self.param_dict)


class LocalJobParametersWindow(QtGui.QWidget):
    """
    Class that handles input of tomopy reconstruction options
    """

    def __init__(self, parameters, parent=None):
        super(LocalJobParametersWindow, self).__init__(parent)
        self.ui = QUiLoader().load('gui/local_job.ui', self)
        self.parameters = parameters
        self.setSystemSpecs()
        self.setDefaultParameters()

    def show(self):
        self.ui.show()
        self.ui.raise_()

    def setSystemSpecs(self):
        memory = psutil.virtual_memory()
        available = '{0:.1f}'.format(round(float(memory.available)/2**30,
                                          decimals=1))
        total = '{0:.1f}'.format(round(float(memory.total)/2**30, decimals=1))
        cores = psutil.cpu_count()
        self.ui.avb_mem_label.setText(available)
        self.ui.tot_mem_label.setText(total)
        self.ui.core_label.setText(str(cores))
        self.ui.core_spinBox.setRange(1, cores)
        self.ui.chunk_spinBox.setMinimum(1)

    def setDefaultParameters(self):
        if self.parameters['ncore'] is None:
            self.ui.core_spinBox.setValue(psutil.cpu_count())
        else:
            self.ui.core_spinBox.setValue(self.parameters['ncore'])

        if self.parameters['nchunk'] is None:
            self.ui.chunk_spinBox.setValue(1)
        else:
            self.ui.chunk_spinBox.setValue(self.parameters['nchunk'])
