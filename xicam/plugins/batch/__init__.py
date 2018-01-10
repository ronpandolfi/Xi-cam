from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals
import os
from functools import partial
from collections import OrderedDict

from PySide import QtGui,QtCore,QtUiTools

from paws.qt.qtapi import QPawsAPI
from paws.qt import widgets
from paws.core.models.ListModel import ListModel
from .. import base
from pipeline import msg
from xicam import config
from xicam import threads
from pyqtgraph import parametertree as pt

class BatchPlugin(base.plugin):
    name = 'Batch'

    def __init__(self, *args, **kwargs):

        # here is a dictionary of available batch workflows for XICAM.
        # note that each batch workflow is composed of two workflows. 
        # one is a processing workflow (written for a single input)
        # and the other is a batch executor which iterates an input 
        # for the processing workflow and optionally harvests its outputs.
        # the names of these two workflows (taken from the paws documentation)
        # are saved here for later use in paws api calls 
        #
        # uris to locate workflow modules from paws:
        self._wf_uris = OrderedDict()
        self._wf_uris['saxs integrator'] = 'XICAM.batch_saxs_integrator'
        self._wf_uris['saxs guinier-porod fitter'] = 'XICAM.batch_saxs_gp_fit'
        # names of the processing workflows:
        self._wf_names = OrderedDict()
        self._wf_names['saxs integrator'] = 'saxs_integrator'
        self._wf_names['saxs guinier-porod fitter'] = 'saxs_gp_fit'
        # names of the batch-execution workflows:
        self._batch_wf_names = OrderedDict()
        self._batch_wf_names['saxs integrator'] = 'batch_saxs_integrator'
        self._batch_wf_names['saxs guinier-porod fitter'] = 'batch_saxs_gp_fit'

        # PawsAPI instances for each workflow,
        # and ParameterTree roots as well
        self._paws = OrderedDict()
        self._root_params = OrderedDict()
        self._run_wf_buttons = OrderedDict()
        # allowed datatypes for ParameterTree entries:
        self.allowed_datatypes = ['int','float','str','bool']
        for wf_title,wf_uri in self._wf_uris.items():
            self._paws[wf_title] = QPawsAPI(QtGui.QApplication.instance())
            self._paws[wf_title].load_packaged_workflow(wf_uri)
            self._add_workflow_params(wf_title)
            # TODO : How should message emissions be handled?
            self._paws[wf_title].set_logmethod(print)
            self._paws[wf_title]._wf_manager.emitMessage.connect(print)
            # Connect the viewer
            #self._paws[wf_title].get_wf(self._wf_names[wf_title]).opFinished.connect( self.update_visuals )

        # create basic ui components
        self.build_ui()
        self.centerwidget = self.viewer_frame
        self.rightwidget = self.wf_frame
        self.bottomwidget = self.batch_control

        # select a default workflow (populates ui content)
        self._current_wf_title = None
        self.select_workflow('saxs integrator')

        #self._current_visual = None
        self._vis_widget = None
        self._placeholder_widget = QtGui.QPlainTextEdit('intentionally blank')

        super(BatchPlugin, self).__init__(*args, **kwargs)

    def build_ui(self):
        self.remove_files_button = QtGui.QPushButton('Remove selected files')
        self.remove_files_button.clicked.connect(self.rm_files)

        self.viewer_frame = QtGui.QFrame()
        self.viewer_layout = QtGui.QGridLayout()
        self.viewer_frame.setLayout(self.viewer_layout)

        self.wf_frame = QtGui.QFrame()
        self.wf_layout = QtGui.QGridLayout()
        self.wf_frame.setLayout(self.wf_layout)
        self.wf_selector = QtGui.QComboBox()
        lm = ListModel(self._wf_uris.keys())
        self.wf_selector.setModel(lm)
        self.wf_selector.currentIndexChanged.connect( partial(self._set_wf_by_idx) )
 
        self.wf_control = pt.ParameterTree()
        self.wf_control.setHeaderLabels(['Operation','Settings'])
        self.wf_layout.addWidget(self.wf_selector,0,0,1,1)
        self.wf_layout.addWidget(self.wf_control,1,0,1,1)
        #self.wf_control.itemClicked.connect(self._display_wf_item)

        self.batch_control = QtGui.QWidget()
        self.batch_list = QtGui.QListWidget()
        self.batch_layout = QtGui.QGridLayout()
        self.batch_control.setLayout(self.batch_layout)
        self.batch_layout.addWidget(self.batch_list,0,0,1,1)
        self.batch_layout.addWidget(self.remove_files_button,1,0,1,1)
        self.batch_list.setSelectionMode(QtGui.QAbstractItemView.ExtendedSelection)

    def _set_wf_by_idx(self,wf_selector_idx):
        wf_title = self.wf_selector.model().list_data()[wf_selector_idx]
        self.select_workflow(wf_title)

    def select_workflow(self,wf_title): 
        self.wf_control.setParameters(
            self._root_params[wf_title],showTop=False)
        self._current_wf_title = wf_title

    # this is an overridden method from base.plugin 
    def openfiles(self, files, operation=None, operationname=None):
        self.batch_list.addItems(files)

    def rm_files(self):
        itms = self.batch_list.selectedItems()
        for itm in itms:
            self.batch_list.takeItem(self.batch_list.row(itm))
 
    def _add_workflow_params(self,wf_title):
        paw = self._paws[wf_title]
        # Create a root parameter for the ParameterTree
        root_param = pt.types.GroupParameter(name='root')
        # Create child parameters for all operations in the processing workflow
        for op_tag in paw.get_wf(self._wf_names[wf_title]).list_op_tags():
            # Connect the ParameterTree
            root_param.addChild(self._op_param(op_tag,wf_title))
        # Create a "run" button as an "ActionParameter"
        # TODO: toggle the text on this run button,
        # and add a signal to "stop"
        run_wf_button=pt.types.ActionParameter(name='&Run')
        run_wf_button.sigActivated.connect( partial(self.toggle_run_wf,wf_title) )
        root_param.addChild(run_wf_button)
        self._root_params[wf_title] = root_param
        self._run_wf_buttons[wf_title] = run_wf_button

    def _op_param(self,op_tag,wf_title):
        paw = self._paws[wf_title]
        # op Parameter 
        p = pt.types.SimpleParameter(   
            name=op_tag, 
            type='bool',
            showTop=False, 
            value=paw.is_op_enabled(op_tag,self._wf_names[wf_title]), 
            expanded=False)
        p.sigValueChanged.connect( partial(self._set_op_enabled,wf_title,op_tag) )
        # child Parameters: inputs and outputs
        op = paw.get_op(op_tag,self._wf_names[wf_title])
        for inp_name in op.inputs.keys():
            inp_val = paw.get_input_setting(op_tag,inp_name,self._wf_names[wf_title]) 
            if op.input_datatype[inp_name] in self.allowed_datatypes:
                p_inp = pt.types.SimpleParameter(
                    name=inp_name,
                    type=op.input_datatype[inp_name],
                    value=inp_val)
            else:
                p_inp = pt.types.SimpleParameter(
                    name=inp_name,
                    type='str',
                    value=str(inp_val))
            p_inp.sigValueChanged.connect( partial(self._set_parameter,wf_title,op_tag,inp_name) )
            p.addChild(p_inp)
            # TODO: Connect paws opChanged signal to some slot 
            # that updates parameters in self.wf_control 
            # for parameters that refer to workflow item inputs 
        for out_name in op.outputs.keys():
            p_out = pt.types.ActionParameter(name='display '+out_name)
            p_out.sigActivated.connect( partial(self._display_op_output,wf_title,op_tag,out_name) )
            p.addChild(p_out)
        return p

    def _display_op_output(self,wf_title,op_name,out_name,param):
        paw = self._paws[wf_title]
        wf = paw.get_wf(self._wf_names[wf_title])
        data = wf.get_op_output(op_name,out_name)
        if data is not None:
            widg = widgets.make_widget(data)
        else:
            widg = self._placeholder_widget
        # clear the output layout
        old_widg_itm = self.viewer_layout.takeAt(0)
        if old_widg_itm is not None:
            if old_widg_itm.widget() is not self._placeholder_widget:
                old_widg_itm.widget().close()
            del old_widg_itm
            #old_widg_itm.widget().deleteLater()
        # set the new widget
        self.viewer_layout.addWidget(widg,0,0)
        #self._current_visual = op_name
        self._vis_widget = widg

    def _set_parameter(self,wf_title,op_name,input_name,param,val):
        paw = self._paws[wf_title]
        paw.set_input(op_name,input_name,val,None,self._wf_names[wf_title])

    def _set_op_enabled(self,wf_title,op_name,param,val):
        paw = self._paws[wf_title]
        if val:
            paw.enable_op(op_name,self._wf_names[wf_title])
        else:
            paw.disable_op(op_name,self._wf_names[wf_title])

    def toggle_run_wf(self,wf_title,param):
        # TODO: toggle the run_wf_button text,
        # and connect signals to stop the workflow once running
        #import xicam.xglobals
        #wfmanager = xicam.xglobals.window.wfmanager

        paw = self._paws[wf_title]
        #paw.select_wf(self._batch_wf_names[wf_title])
        file_list = []
        nfiles = self.batch_list.count()
        for r in range(nfiles):
            p = self.batch_list.item(r).text()
            file_list.append(p)

        # harvest file list and (maybe) PyFAI.AzimuthalIntegrator settings
        # from GUI and Xi-cam internal variables, respectively
        paw.set_input('Batch Execution','input_arrays',[file_list],
            None,self._batch_wf_names[wf_title])
        if wf_title == 'saxs integrator':
            paw.set_input('Integrator Setup','poni_dict',
                config.activeExperiment.getAI().getPyFAI(),
                None,self._batch_wf_names[wf_title])

        #if wfmanager.client is not None:
        #    wfmanager.run_paws(paw)
        #else:
        run_off_thread = threads.method()( paw.execute )
        run_off_thread( self._batch_wf_names[wf_title] )

    #def update_visuals(self,op_tag):
    #    if op_tag == self._current_visual:
    #        if self._vis_widget is self._placeholder_widget:
    #            # visual was previously not ready: try again
    #            self.set_visualizer(op_tag)
    #        else:
    #            visdata = self._get_vis_output(op_tag)
    #            if op_tag in ['Read Image','Integrate to 2d','log(I) 2d']: 
    #                # expect a pyqtgraph.ImageView
    #                self._vis_widget.setImage(visdata) 
    #            elif op_tag in ['Integrate to 1d','log(I) 1d']:
    #                # expect a pyqtgraph.PlotWidget
    #                self._vis_widget.getPlotItem().clear()
    #                self._vis_widget.getPlotItem().plot(visdata)
    #            elif op_tag in ['Output CSV','Output Image']:
    #                # expect a QtGui.QTextEdit
    #                t = widgets.display_text_fast(visdata)
    #                self._vis_widget.setText(t)
    #    self.paw.app.processEvents()

    #def _get_vis_output(self,op_name):
    #    if op_name == 'Read Image':
    #        output_data = self.paw.get_output(op_tag,'image_data',self._wfname)
    #    elif op_tag == 'Integrate to 2d':
    #        output_data = self.paw.get_output(op_tag,'I_at_q_chi',self._wfname)
    #    elif op_tag == 'Integrate to 1d':
    #        output_data = self.paw.get_output(op_tag,'q_I',self._wfname)
    #    elif op_tag == 'log(I) 1d':
    #        output_data = self.paw.get_output(op_tag,'x_logy',self._wfname)
    #    elif op_tag == 'log(I) 2d':
    #        output_data = self.paw.get_output(op_tag,'logx',self._wfname)
    #        # TODO: find a graceful way to handle the nans that result from log(0).
    #        # currently pyqtgraph throws errors about it.
    #    elif op_tag == 'Output CSV':
    #        output_data = self.paw.get_output(op_tag,'file_path',self._wfname)
    #    elif op_tag == 'Output Image':
    #        output_data = self.paw.get_output(op_tag,'file_path',self._wfname)
    #    else:
    #        return '** no visualization data specified **'
    #    return output_data

        # TODO: make some subroutines to clean this section up.
        #root_param.addChild(p)
        #if op_tag == 'Integrate to 1d':
        #    pc = pt.types.SimpleParameter(name='number of q-points',
        #    type='int',value=self.paw.get_input_setting(op_tag,'npt'))
        #    pc.sigValueChanged.connect( partial(self._set_parameter,op_tag,'npt') )  
        #    p.addChild(pc)
        #    pc = pt.types.SimpleParameter(name='polarization factor',
        #    type='float',value=self.paw.get_input_setting(op_tag,'polarization_factor'))
        #    pc.sigValueChanged.connect( partial(self._set_parameter,op_tag,'polarization_factor') )  
        #    p.addChild(pc)
        #elif op_tag == 'Integrate to 2d':
        #    pc = pt.types.SimpleParameter(name='number of q-points',
        #    type='int',value=self.paw.get_input_setting(op_tag,'npt_rad'))
        #    pc.sigValueChanged.connect( partial(self._set_parameter,op_tag,'npt_rad') )  
        #    p.addChild(pc)
        #    pc = pt.types.SimpleParameter(name='number of chi-points',
        #    type='int',value=self.paw.get_input_setting(op_tag,'npt_azim'))
        #    pc.sigValueChanged.connect( partial(self._set_parameter,op_tag,'npt_azim') )  
        #    p.addChild(pc)
        #    pc = pt.types.SimpleParameter(name='polarization factor',
        #    type='float',value=self.paw.get_input_setting(op_tag,'polarization_factor'))
        #    pc.sigValueChanged.connect( partial(self._set_parameter,op_tag,'polarization_factor') )  
        #    p.addChild(pc)
        #elif op_tag in ['Output CSV','Output Image']:
            # TODO: include workflow items in the parametertree,
            # as read-only parameters that display the current value of the input. 
            # TODO: add a browse button for dir_path.
            # if dir_path is set manually, edit the workflow
            # so that dir_path is no longer set as a workflow item. 
            # TODO: add a text entry field for filename.
            # if filename is set manually, edit the workflow
            # so that filename is no longer set as a workflow item.
        #    pc = pt.types.SimpleParameter(name='file tag',
        #    type='str',value=self.paw.get_input_setting(op_tag,'filetag'))
        #    pc.sigValueChanged.connect( partial(self._set_parameter,op_tag,'filetag') )  
        #    p.addChild(pc)
        #    if op_tag == 'Output Image':
        #        pc = pt.types.SimpleParameter(name='extension',
        #        type='str',value=self.paw.get_input_setting(op_tag,'ext'))
        #        pc.sigValueChanged.connect( partial(self._set_parameter,op_tag,'ext') )  
        #        p.addChild(pc)
        #        pc = pt.types.SimpleParameter(name='overwrite',
        #        type='bool',value=self.paw.get_input_setting(op_tag,'overwrite'))
        #        pc.sigValueChanged.connect( partial(self._set_parameter,op_tag,'overwrite') )  
        #        p.addChild(pc)


