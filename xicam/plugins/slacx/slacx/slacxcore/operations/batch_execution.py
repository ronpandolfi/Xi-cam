import re
from collections import OrderedDict
import glob

from slacxop import Batch
import optools

class BatchFromFiles(Batch):
    """
    Provides a sequence of inputs to be used in repeated execution of a workflow.
    Collects the outputs produced for each of the inputs.
    """

    def __init__(self):
        input_names = ['dir_path','regex','input_route','saved_ops']
        output_names = ['batch_inputs','batch_outputs']
        super(BatchFromFiles,self).__init__(input_names,output_names)
        self.input_doc['dir_path'] = 'path to directory containing batch of files to be used as input'
        self.input_doc['regex'] = 'string with * wildcards that will be substituted to indicate input files'
        self.input_doc['input_route'] = 'inputs constructed by the batch executor are directed to this uri'
        self.input_doc['saved_ops'] = 'list of ops to be saved in the batch_outputs: default all ops'
        self.output_doc['batch_inputs'] = 'list of dicts of [input_route:input_value]'
        self.output_doc['batch_outputs'] = 'list of dicts of [output_route:output_value]'
        self.categories = ['EXECUTION.BATCH']
        self.input_src['dir_path'] = optools.fs_input
        self.input_src['regex'] = optools.user_input 
        self.input_type['regex'] = optools.str_type
        self.inputs['regex'] = '*.tif' 
        self.input_src['input_route'] = optools.wf_input 
        self.input_src['saved_ops'] = optools.wf_input 
        self.input_type['saved_ops'] = optools.list_type 
        self.inputs['saved_ops'] = []
        
    def run(self):
        """
        For Batch, this should build a list of [uri:value] dicts to be used in the workflow.
        """
        dirpath = self.inputs['dir_path']
        rx = self.inputs['regex']
        inproute = self.inputs['input_route']
        #batch_list = [dirpath+'/'+rx.replace('*',sub) for sub in subs]
        batch_list = glob.glob(dirpath+'/'+rx)
        input_dict_list = []
        output_dict_list = []
        for filename in batch_list:
            inp_dict = OrderedDict() 
            inp_dict[inproute] = filename
            input_dict_list.append(inp_dict)
            output_dict_list.append(OrderedDict())
        self.outputs['batch_inputs'] = input_dict_list
        # Instantiate the batch_outputs list
        self.outputs['batch_outputs'] = output_dict_list 

    def input_list(self):
        return self.outputs['batch_inputs']

    def output_list(self):
        return self.outputs['batch_outputs']

    def input_routes(self):
        """Use the Batch.input_locator to list uri's of all input routes"""
        return [ self.input_locator['input_route'].val ]

    def saved_ops(self):
        """Use the Batch.input_locator to list uri's of ops to be saved/stored after execution"""
        return list(self.input_locator['saved_ops'].val)



