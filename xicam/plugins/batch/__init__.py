from __future__ import absolute_import
from __future__ import unicode_literals
import os
from functools import partial
from collections import OrderedDict

from PySide.QtGui import *
from PySide.QtCore import *
from PySide import QtUiTools

from paws.api import PawsAPI
from paws.ui import uitools
from paws.ui import widgets
from paws.core import pawstools
from paws.core.operations import Operation as opmod
from .. import base
from pipeline import msg
from xicam import config
from pyqtgraph import parametertree as pt



class EnableGroupParameterItem(pt.types.ParameterItem):
    def __init__(self,*args,**kwargs):
        super(EnableGroupParameterItem, self).__init__(*args,**kwargs)
        self.addWidget(QCheckBox())

class EnableGroupParameter(pt.Parameter):
    itemClass = EnableGroupParameterItem


class BatchPlugin(base.plugin):
    name = 'Batch'

    def __init__(self, *args, **kwargs):

        self.paw = PawsAPI()
        self._wfname = 'img_process'
        self._batch_wfname = 'batch'
        #self.pawswidget = BatchWidget.BatchWidget(self.paw)
        
        self.ops = OrderedDict()
        self.ops['Read Image'] = 'IO.IMAGE.FabIOOpen'
        self.ops['Integrate to 2d'] = 'PROCESSING.INTEGRATION.ApplyIntegrator2d'
        self.ops['Integrate to 1d'] = 'PROCESSING.INTEGRATION.ApplyIntegrator1d'
        self.ops['log(I) 2d'] = 'PROCESSING.BASIC.ArrayLog'
        self.ops['log(I) 1d'] = 'PROCESSING.BASIC.LogY'
        self.ops['Output CSV'] = 'IO.CSV.WriteArrayCSV'
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
        self.remove_files_button = QPushButton('Remove selected files')
        self.remove_files_button.clicked.connect(self.rm_files)

        self.viewer_tabs = QStackedWidget()
        self.wf_control = pt.ParameterTree()
        self.wf_control.setHeaderLabels(['Operation','Enabled'])
        self.wf_control.itemSelectionChanged.connect(self.itemSelectionChanged)

        self.batch_control = QWidget()
        self.batch_list = QListWidget()
        self.batch_layout = QGridLayout()
        self.batch_control.setLayout(self.batch_layout)
        self.batch_layout.addWidget(self.batch_list,0,0,1,1)
        self.batch_layout.addWidget(self.remove_files_button,1,0,1,1)

        self.batch_list.setSelectionMode(QAbstractItemView.ExtendedSelection)

    def itemSelectionChanged(self,*args,**kwargs):
        item=self.wf_control.selectedItems()[0]
        while item.depth>0: item=item.param.parent().items.keys()[0] # ascend until at 'operation' depth
        if item.param.name()=='&Run': return # do nothing if its the run button
        selected_tag = item.param.name()
        self.set_visualizer(selected_tag,True)


    def rm_files(self):
        itms = self.batch_list.selectedItems()
        for itm in itms:
            self.batch_list.takeItem(self.batch_list.row(itm))

    def wf_setup(self):
        self.paw.add_wf(self._wfname)
        self.paw.connect_wf_input('image_path','Read Image.inputs.path',self._wfname)
        self.paw.add_wf(self._batch_wfname)

        # Set up the batch execution Operation first
        self.paw.select_wf(self._batch_wfname)
        self.paw.add_op('batch','EXECUTION.BATCH.BatchFromFiles')
        self.paw.set_input('batch','workflow',self._wfname)
        self.paw.set_input('batch','input_name','image_path')

        # Set up the rest of the workflow
        self.paw.select_wf(self._wfname)
        root = pt.types.GroupParameter(name='root')
        for op_tag,op_uri in self.ops.items():
            p = pt.types.SimpleParameter(name=op_tag, type='bool',showTop=False, value=True, expanded=False)
            root.addChild(p)

            # TODO: add operation parameters as children so they are editable
            # If an op has children, it becomes expandable
            p.addChild(pt.types.SimpleParameter(name='Example',type='int',value='42'))

            # Add the op to the workflow
            self.paw.add_op(op_tag,op_uri,self._wfname)

            # Set up the inputs....
            self._default_op_setup(op_tag)

        run_wf_button=pt.types.ActionParameter(name='&Run')
        run_wf_button.sigActivated.connect(self.run_wf)
        root.addChild(run_wf_button)

        self.wf_control.setParameters(root,showTop=False)

    def toggle_enabled(self,op_tag,state):
        if not state == 0:
            self.paw.enable_op(op_tag,self._wfname)
        else:
            self.paw.disable_op(op_tag,self._wfname)

    def edit_op(self,op_tag):
        pass

    def _default_op_setup(self, op_tag):
        if op_tag == 'Read Image':
            # This is where the batch operation will set its inputs
            self.paw.set_input(op_tag,'path','')

        elif op_tag == 'Integrate to 1d' or op_tag == 'Integrate to 2d':
            self.paw.set_input(op_tag,'data','Read Image.outputs.image_data')
            self.paw.set_input(op_tag,'integrator',config.activeExperiment.getAI(),'object')

        elif op_tag == 'log(I) 1d':
            self.paw.set_input(op_tag,'x_y','Integrate to 1d.outputs.q_I')

        elif op_tag == 'log(I) 2d':
            self.paw.set_input(op_tag,'x','Integrate to 2d.outputs.I_at_q_chi')

        elif op_tag == 'Output CSV':
            self.paw.set_input(op_tag,'array','Integrate to 1d.outputs.q_I')
            self.paw.set_input(op_tag,'headers',['q','I'])
            self.paw.set_input(op_tag,'dir_path','Read Image.outputs.dir_path','workflow item')
            self.paw.set_input(op_tag,'filename','Read Image.outputs.filename','workflow item')
            self.paw.set_input(op_tag,'filetag','_processed')

        elif op_tag == 'Output Image':
            self.paw.set_input(op_tag,'image_data','Integrate to 2d.outputs.I_at_q_chi')
            self.paw.set_input(op_tag,'dir_path','Read Image.outputs.dir_path','workflow item')
            self.paw.set_input(op_tag,'filename','Read Image.outputs.filename','workflow item')
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
                tab_idx = self.viewer_tabs.addWidget(widg)
            self.viewer_tabs.setCurrentWidget(widg)
        else:
            widg = self.output_widgets.pop(op_tag)
            if widg is not None:
                tab_idx = self.viewer_tabs.indexOf(widg)
                widg.close()
                if not tab_idx == -1:
                    self.viewer_tabs.removeTab(tab_idx)

    def make_widget(self,op_tag):
        if op_tag == 'Read Image':
            output_data = self.paw.get_output(op_tag,'image_data',self._wfname)
        elif op_tag == 'Integrate to 2d':
            output_data = self.paw.get_output(op_tag,'I',self._wfname)
        elif op_tag == 'Integrate to 1d':
            output_data = self.paw.get_output(op_tag,'q_I',self._wfname)
        elif op_tag == 'log(I) 1d':
            output_data = self.paw.get_output(op_tag,'x_logy',self._wfname)
        elif op_tag == 'log(I) 2d':
            output_data = self.paw.get_output(op_tag,'logx',self._wfname)
        elif op_tag == 'Output CSV':
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
            if isinstance(widg,QWidget):
                widg.repaint()

    def openfiles(self, files, operation=None, operationname=None):
        self.batch_list.addItems(files)





