import re

from slacxop import Operation
import optools

class BatchFromFiles(Operation):
    """
    Provides a sequence of inputs to be used in repeated execution of a workflow.
    Collects the outputs produced for each of the inputs in the sequence.
    """

    def __init__(self):
        input_names = ['dir_path','regex','substitutions']
        output_names = ['batch_iterator','batch_outputs']
        super(BatchFromFiles,self).__init__(input_names,output_names)
        self.input_doc['dir_path'] = 'filesystem path pointing to directory containing the batch of files to be used as input'
        self.input_doc['regex'] = 'a string with * wildcards that will be sequentially substituted with the provided substitutions'
        self.input_doc['substitutions'] = 'an array whose elements will be substituted into regex to indicate elements in the batch'
        self.output_doc['batch_iterator'] = 'Batch returns an iterator that emits string file paths to be used as inputs to a workflow one at a time'
        self.output_doc['batch_outputs'] = 'Dict keyed by batch_iterator items. Each value in the dict is itself a dict containing the workflow outputs for that particular input.'
        self.categories = ['EXECUTION.BATCH']
        self.input_src['dir_path'] = optools.fs_input
        self.input_src['regex'] = optools.text_input 
        self.input_src['substitutions'] = optools.op_input 
        self.input_type['regex'] = optools.str_type
        
    def run(self):
        """
        For Batch, this should form a list of inputs to be used in the workflow,
        then wrap that list in an iterator and return that iterator.
        """
        # Optional- create references in the local namespace for cleaner code.
        dirpath = self.inputs['dir_path']
        rx = self.inputs['regex']
        subs = self.inputs['substitutions']
        # Perform the computation
        batch_list = [dirpath+re.sub('*',sub,rx) for sub in subs]
        # Save the output
        self.outputs['batch_iterator'] = iter(batch_list) 
        # Instantiate the batch_output dict
        self.outputs['batch_outputs'] = dict.fromkeys(batch_list)

