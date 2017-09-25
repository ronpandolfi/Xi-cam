from __future__ import absolute_import
from __future__ import unicode_literals
import os
from functools import partial
from collections import OrderedDict

from PySide import QtGui,QtCore,QtUiTools

from paws.qt.qtapi import QPawsAPI
from paws.ui import uitools
from paws.ui import widgets
from paws.core import pawstools
from paws.core.operations import Operation as opmod
from .. import base
from pipeline import msg
from xicam import config
from xicam import threads
from pyqtgraph import parametertree as pt


class BatchPlugin(base.plugin):
    name = 'Batch'

    def __init__(self, *args, **kwargs):

        self.paw = QPawsAPI(QtGui.QApplication.instance())
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
        self.centerwidget = self.viewer_frame
        self.rightwidget = self.wf_control
        self.bottomwidget = self.batch_control

        self.wf_setup()

        self._current_visual = None
        self._vis_widget = None
        self._placeholder_widget = QtGui.QPlainTextEdit('** no output data **')

        super(BatchPlugin, self).__init__(*args, **kwargs)

    def build_ui(self):
        self.remove_files_button = QtGui.QPushButton('Remove selected files')
        self.remove_files_button.clicked.connect(self.rm_files)

        self.viewer_frame = QtGui.QFrame()
        self.viewer_layout = QtGui.QGridLayout()
        self.viewer_frame.setLayout(self.viewer_layout)
        
        self.wf_control = pt.ParameterTree()
        self.wf_control.setHeaderLabels(['Operation','Settings'])
        self.wf_control.itemClicked.connect(self._display_op_output)

        self.batch_control = QtGui.QWidget()
        self.batch_list = QtGui.QListWidget()
        self.batch_layout = QtGui.QGridLayout()
        self.batch_control.setLayout(self.batch_layout)
        self.batch_layout.addWidget(self.batch_list,0,0,1,1)
        self.batch_layout.addWidget(self.remove_files_button,1,0,1,1)

        self.batch_list.setSelectionMode(QtGui.QAbstractItemView.ExtendedSelection)

    def _display_op_output(self):
        sel = self.wf_control.selectedItems()
        if any(sel):
            itm = sel[0]
            while itm.depth>0: itm=itm.param.parent().items.keys()[0] # ascend until at 'operation' depth
            itm_tag = itm.param.name()
            if itm_tag in self.ops.keys() and not itm_tag=='&Run':
                self.set_visualizer(itm_tag)

    def rm_files(self):
        itms = self.batch_list.selectedItems()
        for itm in itms:
            self.batch_list.takeItem(self.batch_list.row(itm))

    def wf_setup(self):
        self.paw.add_wf(self._wfname)
        self.paw.add_wf_input('image_path','Read Image.inputs.path',self._wfname)
        self.paw.add_wf(self._batch_wfname)

        # Set up the batch execution Operation first
        self.paw.select_wf(self._batch_wfname)
        self.paw.add_op('batch','EXECUTION.BATCH.BatchFromFiles')
        self.paw.set_input('batch','workflow',self._wfname)
        self.paw.set_input('batch','input_name','image_path')

        # Set up the read-and-integrate workflow
        self.paw.select_wf(self._wfname)
        # Create a root parameter for the ParameterTree
        root_param = pt.types.GroupParameter(name='root')
        for op_tag,op_uri in self.ops.items():
            # Add the op to the workflow
            self.paw.add_op(op_tag,op_uri,self._wfname)
            # Set up the inputs
            self._default_op_setup(op_tag)
            # Connect the ParameterTree
            self._param_setup(root_param,op_tag)
        # Connect the viewer
        self.paw.get_wf(self._wfname).opFinished.connect( self.update_visuals )
        run_wf_button=pt.types.ActionParameter(name='&Run')
        run_wf_button.sigActivated.connect(self.run_wf)
        root_param.addChild(run_wf_button)
        self.wf_control.setParameters(root_param,showTop=False)
        # TODO: Connect paws opChanged signal to some signal 
        # that updates parameters in self.wf_control 
        # for parameters that refer to workflow item inputs 

    def _param_setup(self,root_param,op_tag):
        # op Parameter 
        default_enabled = True
        if op_tag in ['Output CSV','Output Image']:
            default_enabled = False 
            self.paw.disable_op(op_tag,self._wfname)
        p = pt.types.SimpleParameter(name=op_tag, type='bool',showTop=False, value=default_enabled, expanded=False)
        p.sigValueChanged.connect( partial(self._set_op_enabled,op_tag) )
        # child Parameters
        if op_tag == 'Integrate to 1d':
            pc = pt.types.SimpleParameter(name='number of q-points',
            type='int',value=self.paw.get_input_setting(op_tag,'npt'))
            pc.sigValueChanged.connect( partial(self._set_parameter,op_tag,'npt') )  
            p.addChild(pc)
            pc = pt.types.SimpleParameter(name='polarization factor',
            type='float',value=self.paw.get_input_setting(op_tag,'polarization_factor'))
            pc.sigValueChanged.connect( partial(self._set_parameter,op_tag,'polarization_factor') )  
            p.addChild(pc)
        elif op_tag == 'Integrate to 2d':
            pc = pt.types.SimpleParameter(name='number of q-points',
            type='int',value=self.paw.get_input_setting(op_tag,'npt_rad'))
            pc.sigValueChanged.connect( partial(self._set_parameter,op_tag,'npt_rad') )  
            p.addChild(pc)
            pc = pt.types.SimpleParameter(name='number of chi-points',
            type='int',value=self.paw.get_input_setting(op_tag,'npt_azim'))
            pc.sigValueChanged.connect( partial(self._set_parameter,op_tag,'npt_azim') )  
            p.addChild(pc)
            pc = pt.types.SimpleParameter(name='polarization factor',
            type='float',value=self.paw.get_input_setting(op_tag,'polarization_factor'))
            pc.sigValueChanged.connect( partial(self._set_parameter,op_tag,'polarization_factor') )  
            p.addChild(pc)
        elif op_tag in ['Output CSV','Output Image']:
            # TODO: include workflow items in the parametertree,
            # as read-only parameters that display the current value of the input. 
            # TODO: add a browse button for dir_path.
            # if dir_path is set manually, edit the workflow
            # so that dir_path is no longer set as a workflow item. 
            # TODO: add a text entry field for filename.
            # if filename is set manually, edit the workflow
            # so that filename is no longer set as a workflow item.
            pc = pt.types.SimpleParameter(name='file tag',
            type='str',value=self.paw.get_input_setting(op_tag,'filetag'))
            pc.sigValueChanged.connect( partial(self._set_parameter,op_tag,'filetag') )  
            p.addChild(pc)
            if op_tag == 'Output Image':
                pc = pt.types.SimpleParameter(name='extension',
                type='str',value=self.paw.get_input_setting(op_tag,'ext'))
                pc.sigValueChanged.connect( partial(self._set_parameter,op_tag,'ext') )  
                p.addChild(pc)
                pc = pt.types.SimpleParameter(name='overwrite',
                type='bool',value=self.paw.get_input_setting(op_tag,'overwrite'))
                pc.sigValueChanged.connect( partial(self._set_parameter,op_tag,'overwrite') )  
                p.addChild(pc)
        root_param.addChild(p)

    def _set_parameter(self,op_tag,input_name,param,val):
        self.paw.set_input(op_tag,input_name,val)

    def _set_op_enabled(self,op_tag,param,val):
        if val:
            self.paw.enable_op(op_tag,self._wfname)
        else:
            self.paw.disable_op(op_tag,self._wfname)

    def _default_op_setup(self,op_tag):
        if op_tag == 'Read Image':
            # This is where the batch operation will set its inputs
            self.paw.set_input(op_tag,'path','')
        elif op_tag == 'Integrate to 1d' or op_tag == 'Integrate to 2d':
            self.paw.set_input(op_tag,'data','Read Image.outputs.image_data')
            self.paw.set_input(op_tag,'integrator',config.activeExperiment.getAI(),'auto')
        elif op_tag == 'log(I) 1d':
            self.paw.set_input(op_tag,'x_y','Integrate to 1d.outputs.q_I')
        elif op_tag == 'log(I) 2d':
            self.paw.set_input(op_tag,'x','Integrate to 2d.outputs.I_at_q_chi')
        elif op_tag == 'Output CSV':
            self.paw.set_input(op_tag,'array','Integrate to 1d.outputs.q_I')
            self.paw.set_input(op_tag,'headers',['q','I'])
            self.paw.set_input(op_tag,'dir_path','Read Image.outputs.dir_path','workflow item')
            self.paw.set_input(op_tag,'filename','Read Image.outputs.filename','workflow item')
            self.paw.set_input(op_tag,'filetag','_q_I')
        elif op_tag == 'Output Image':
            self.paw.set_input(op_tag,'image_data','Integrate to 2d.outputs.I_at_q_chi')
            self.paw.set_input(op_tag,'dir_path','Read Image.outputs.dir_path','workflow item')
            self.paw.set_input(op_tag,'filename','Read Image.outputs.filename','workflow item')
            self.paw.set_input(op_tag,'filetag','_q_chi_I')
            self.paw.set_input(op_tag,'overwrite',True)
            self.paw.set_input(op_tag,'ext','.edf')

    def toggle_enabled(self,op_tag,state):
        if not state == 0:
            self.paw.enable_op(op_tag,self._wfname)
        else:
            self.paw.disable_op(op_tag,self._wfname)

    def edit_op(self,op_tag):
        pass

    def set_visualizer(self,op_tag):
        widg = self.make_widget(op_tag)
        # clear the output layout
        old_widg_itm = self.viewer_layout.takeAt(0)
        if old_widg_itm is not None:
            if old_widg_itm.widget() is not self._placeholder_widget:
                old_widg_itm.widget().close()
            del old_widg_itm
            #old_widg_itm.widget().deleteLater()
        # set the new widget
        self.viewer_layout.addWidget(widg,0,0)
        self._current_visual = op_tag
        self._vis_widget = widg

    def _get_vis_output(self,op_tag):
        if op_tag == 'Read Image':
            output_data = self.paw.get_output(op_tag,'image_data',self._wfname)
        elif op_tag == 'Integrate to 2d':
            output_data = self.paw.get_output(op_tag,'I_at_q_chi',self._wfname)
        elif op_tag == 'Integrate to 1d':
            output_data = self.paw.get_output(op_tag,'q_I',self._wfname)
        elif op_tag == 'log(I) 1d':
            output_data = self.paw.get_output(op_tag,'x_logy',self._wfname)
        elif op_tag == 'log(I) 2d':
            output_data = self.paw.get_output(op_tag,'logx',self._wfname)
            # TODO: find a graceful way to handle the nans that result from log(0).
            # currently pyqtgraph throws errors about it.
        elif op_tag == 'Output CSV':
            output_data = self.paw.get_output(op_tag,'file_path',self._wfname)
        elif op_tag == 'Output Image':
            output_data = self.paw.get_output(op_tag,'file_path',self._wfname)
        else:
            return '** no visualization data specified **'
        return output_data

    def make_widget(self,op_tag):
        output_data = self._get_vis_output(op_tag)
        # Form a widget from the output data 
        if output_data is not None:
            widg = widgets.make_widget(output_data)
        else:
            widg = self._placeholder_widget 
        return widg

    def run_wf(self):
        self.paw.select_wf(self._batch_wfname)
        file_list = []
        nfiles = self.batch_list.count()
        for r in range(nfiles):
            p = self.batch_list.item(r).text()
            file_list.append(p)
        self.paw.set_input('batch','file_list',file_list)
        # TODO: A way to stop the workflow once running
        # TODO: Ensure this is thread-safe
        run_off_thread = threads.method()(self.paw.execute) 
        run_off_thread()
        #self.paw.execute()
        #self.update_visuals()

    def update_visuals(self,op_tag):
        if op_tag == self._current_visual:
            if self._vis_widget is self._placeholder_widget:
                # visual was previously not ready: try again
                self.set_visualizer(op_tag)
            else:
                visdata = self._get_vis_output(op_tag)
                if op_tag in ['Read Image','Integrate to 2d','log(I) 2d']: 
                    # expect a pyqtgraph.ImageView
                    self._vis_widget.setImage(visdata) 
                elif op_tag in ['Integrate to 1d','log(I) 1d']:
                    # expect a pyqtgraph.PlotWidget
                    self._vis_widget.getPlotItem().clear()
                    self._vis_widget.getPlotItem().plot(visdata)
                elif op_tag in ['Output CSV','Output Image']:
                    # expect a QtGui.QTextEdit
                    t = widgets.display_text_fast(visdata)
                    self._vis_widget.setText(t)
        self.paw.app.processEvents()

    def openfiles(self, files, operation=None, operationname=None):
        self.batch_list.addItems(files)



