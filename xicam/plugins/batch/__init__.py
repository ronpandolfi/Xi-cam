from __future__ import absolute_import
from __future__ import unicode_literals
import os
from functools import partial
from collections import OrderedDict

from PySide import QtGui, QtCore, QtUiTools

from paws.api import PawsAPI
from paws.ui import uitools
from paws.ui import widgets
from paws.core import pawstools
from paws.core.operations import Operation as opmod
from .. import base
from pipeline import msg
from xicam import config

class BatchPlugin(base.plugin):
    name = 'Batch'

    def __init__(self, *args, **kwargs):

        self.paw = PawsAPI()
        self._wfname = 'img_process'
        self._batch_wfname = 'batch'
        #self.pawswidget = BatchWidget.BatchWidget(self.paw)
        
        self.ops = OrderedDict()
        self.ops['read_image'] = 'IO.IMAGE.FabIOOpen'
        self.ops['calibrate_image'] = 'PROCESSING.INTEGRATION.ApplyIntegrator2d'
        self.ops['integrate_image'] = 'PROCESSING.INTEGRATION.ApplyIntegrator1d'
        self.ops['log_I_2d'] = 'PROCESSING.BASIC.ArrayLog'
        self.ops['log_I_1d'] = 'PROCESSING.BASIC.LogY'
        self.ops['write_csv'] = 'IO.CSV.WriteArrayCSV'
        self.ops['Output Image'] = 'IO.IMAGE.FabIOWrite'

        for nm,opuri in self.ops.items():
            self.paw.activate_op(opuri)       
        self.paw.activate_op('EXECUTION.BATCH.BatchFromFiles')

        self.build_ui()
        self.centerwidget = self.viewer_tabs
        self.rightwidget = self.wf_control
        self.bottomwidget = self.batch_control

        self.wf_setup()
        self.output_widgets = {} 

        super(BatchPlugin, self).__init__(*args, **kwargs)

    def build_ui(self):
        #self.add_files_button.setText('Add selected files')
        #self.add_files_button.clicked.connect(self.add_files)
        self.remove_files_button = QtGui.QPushButton('Remove selected files')
        self.remove_files_button.clicked.connect(self.rm_files)

        self.viewer_tabs = QtGui.QTabWidget()
        self.viewer_tabs.clear()
        self.wf_control = QtGui.QGroupBox('workflow control')
        self.wf_layout = QtGui.QGridLayout()
        self.wf_control.setLayout(self.wf_layout)

        self.batch_control = QtGui.QWidget()
        self.batch_list = QtGui.QListWidget()
        self.batch_layout = QtGui.QGridLayout()
        self.batch_control.setLayout(self.batch_layout)
        self.batch_layout.addWidget(self.batch_list,0,0,1,1)
        self.batch_layout.addWidget(self.remove_files_button,1,0,1,1)

        self.batch_list.setSelectionMode(QtGui.QAbstractItemView.ExtendedSelection)


    def rm_files(self):
        itms = self.batch_list.selectedItems()
        for itm in itms:
            self.batch_list.takeItem(self.batch_list.row(itm))

    def wf_setup(self):
        self.paw.add_wf(self._wfname)
        self.paw.connect_wf_input('image_path','read_image.inputs.path',self._wfname)
        self.paw.add_wf(self._batch_wfname)

        # Set up the batch execution Operation first
        self.paw.select_wf(self._batch_wfname)
        self.paw.add_op('batch','EXECUTION.BATCH.BatchFromFiles')
        self.paw.set_input('batch','workflow',self._wfname)
        self.paw.set_input('batch','input_name','image_path')

        # Set up the rest of the workflow
        self.paw.select_wf(self._wfname)
        for op_tag,op_uri in self.ops.items():
            add_op_row = self.paw.op_count(self._wfname)+1

            # Add the op to the workflow
            self.paw.add_op(op_tag,op_uri,self._wfname)
            # Set up the inputs....
            self._default_op_setup(op_tag)

            # Add the op name in a pushbutton 
            op_button = QtGui.QPushButton(op_tag)
            op_button.clicked.connect( partial(self.edit_op,op_tag) )
            self.wf_layout.addWidget(op_button,add_op_row,0,1,1)


            # Add buttons to interact with op 
            #op_edit_button = QtGui.QPushButton('edit')
            #op_edit_button.clicked.connect( partial(self.edit_op,op_tag,op_name) )
            enable_toggle = QtGui.QCheckBox('enable')
            enable_toggle.setCheckState(QtCore.Qt.Checked)
            enable_toggle.stateChanged.connect( partial(self.toggle_enabled,op_tag) )
            vis_toggle = QtGui.QCheckBox('view')
            vis_toggle.stateChanged.connect( partial(self.set_visualizer,op_tag) )
            self.wf_layout.addWidget(enable_toggle,add_op_row,3,1,1)
            self.wf_layout.addWidget(vis_toggle,add_op_row,4,1,1)
            #self.wf_layout.addWidget(op_edit_button,add_op_row,5,1,1)
        self.run_wf_button = QtGui.QPushButton('&Run')
        self.run_wf_button.clicked.connect(self.run_wf) 
        self.wf_layout.addWidget(self.run_wf_button,self.paw.op_count(self._wfname)+1,0,1,3)

    def toggle_enabled(self,op_tag,state):
        if not state == 0:
            self.paw.enable_op(op_tag,self._wfname)
        else:
            self.paw.disable_op(op_tag,self._wfname)

    def edit_op(self,op_tag):
        pass

    def _default_op_setup(self, op_tag):
        if op_tag == 'read_image':
            # This is where the batch operation will set its inputs
            self.paw.set_input(op_tag,'path','')

        elif op_tag == 'calibrate_image' or op_tag == 'integrate_image':
            self.paw.set_input(op_tag,'data','read_image.outputs.image_data')
            self.paw.set_input(op_tag,'integrator',config.activeExperiment.getAI(),'object')

        elif op_tag == 'log_I_1d':
            self.paw.set_input(op_tag,'x_y','integrate_image.outputs.q_I')

        elif op_tag == 'log_I_2d':
            self.paw.set_input(op_tag,'x','calibrate_image.outputs.I')

        elif op_tag == 'write_csv':
            self.paw.set_input(op_tag,'array','integrate_image.outputs.q_I')
            self.paw.set_input(op_tag,'headers',['q','I'])
            self.paw.set_input(op_tag,'dir_path','read_image.outputs.dir_path','workflow item')
            self.paw.set_input(op_tag,'filename','read_image.outputs.filename')
            self.paw.set_input(op_tag,'filetag','_processed')

        elif op_tag == 'Output Image':
            self.paw.set_input(op_tag,'image_data','calibrate_image.outputs.I')
            self.paw.set_input(op_tag,'dir_path','read_image.outputs.dir_path')
            self.paw.set_input(op_tag,'filename','read_image.outputs.filename')
            self.paw.set_input(op_tag,'suffix','_processed')
            self.paw.set_input(op_tag,'ext','.edf')


    def set_visualizer(self,op_tag,state):
        if not state==0:
            # Find, create, or otherwise open the widget
            if not op_tag in self.output_widgets.keys():
                widg = self.make_widget(op_tag)
                self.output_widgets[op_tag] = widg
            else:
                # The user closed the tab
                # instead of un-checking the visualizer box,
                # so the widget should still be in self.output_widgets
                widg = self.output_widgets[op_tag]

            if self.viewer_tabs.indexOf(widg) == -1:
                tab_idx = self.viewer_tabs.addTab(widg,op_tag)
            self.viewer_tabs.setCurrentWidget(widg)
        else:
            widg = self.output_widgets.pop(op_tag)
            if widg is not None:
                tab_idx = self.viewer_tabs.indexOf(widg)
                widg.close()
                if not tab_idx == -1:
                    self.viewer_tabs.removeTab(tab_idx)

    def make_widget(self,op_tag):
        if op_tag == 'read_image':
            output_data = self.paw.get_output(op_tag,'image_data',self._wfname)
        elif op_tag == 'calibrate_image':
            output_data = self.paw.get_output(op_tag,'I',self._wfname)
        elif op_tag == 'integrate_image':
            output_data = self.paw.get_output(op_tag,'q_I',self._wfname)
        elif op_tag == 'log_I_1d':
            output_data = self.paw.get_output(op_tag,'x_logy',self._wfname)
        elif op_tag == 'log_I_2d':
            output_data = self.paw.get_output(op_tag,'logx',self._wfname)
        elif op_tag == 'write_csv':
            output_data = self.paw.get_output(op_tag,'csv_path',self._wfname)
        elif op_tag == 'Output Image':
            output_data = self.paw.get_output(op_tag,'file_path',self._wfname)
        # Form a widget from the output data 
        widg = widgets.make_widget(output_data)
        return widg

    def run_wf(self):
        self.paw.select_wf(self._batch_wfname)
        file_list = []
        nfiles = self.batch_list.count()
        for r in range(nfiles):
            p = self.batch_list.item(r).text()
            file_list.append(p)
        self.paw.set_input('batch','file_list',file_list)

        self.paw.execute()

        self.update_visuals()

    def update_visuals(self):
        for widg in self.output_widgets:
            if isinstance(widg,QtGui.QWidget):
                widg.repaint()

    def openfiles(self, files, operation=None, operationname=None):
        self.batch_list.addItems(files)





