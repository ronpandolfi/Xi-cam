from slacxop import Operation
import optools


class selectBatchItemListFromOps(Operation):
    """Return list of outputs for a given variable from batch."""

    def __init__(self):
        input_names = ['batch_outputs','operation','from_outputs','var_name']
        output_names = ['var_list']
        super(selectBatchItemListFromOps, self).__init__(input_names, output_names)
        self.input_doc['batch_outputs'] = 'From the desired batch, select outputs -> batch_outputs in the workflow'
        self.input_doc['operation'] = 'Name of the operation whose items you want, as it appears in your workflow'
        self.input_doc['from_outputs'] = "True if you want an operation's outputs, False if you want its inputs"
        self.input_doc['var_name'] = 'Name of the variable you want'
        self.output_doc['var_list'] = 'A list, ordered by index in batch, of the selected variable'
        # source & type
        self.input_src['batch_outputs'] = optools.wf_input
        self.input_src['from_outputs'] = optools.user_input
        self.input_src['operation'] = optools.wf_input
        self.input_src['var_name'] = optools.user_input
        self.input_type['from_outputs'] = optools.bool_type
        self.input_type['operation'] = optools.str_type
        self.input_type['var_name'] = optools.str_type
        # defaults
        self.inputs['from_outputs'] = True
        self.categories = ['MISC.SLACX OBJECT MANIPULATION']

    def run(self):
        batch_outputs = self.inputs['batch_outputs']
        operation = self.inputs['operation']
        var_name = self.inputs['var_name']
        num_items = len(batch_outputs)
        var_list = []
        for ii in range(num_items):
            if self.inputs['from_outputs'] == True:
                item = batch_outputs[ii][operation].outputs[var_name]
            else:
                item = batch_outputs[ii][operation].inputs[var_name]
            var_list.append(item)
        self.outputs['var_list'] = var_list

'''
class selectBatchItems2(Operation):
    """Return list of outputs for a given variable from batch."""

    def __init__(self):
        input_names = ['batch_outputs','operation','from_outputs','var_name']
        output_names = ['var_list']
        super(selectBatchItems2, self).__init__(input_names, output_names)
        self.input_doc['batch_outputs'] = 'From the desired batch, select outputs -> batch_outputs in the workflow'
        self.input_doc['operation'] = 'Operation whose items you want, selected in your workflow'
        self.input_doc['from_outputs'] = "True if you want an operation's outputs, False if you want its inputs"
        self.input_doc['var_name'] = 'Name of the variable you want'
        self.output_doc['var_list'] = 'A list, ordered by index in batch, of the selected variable'
        # source & type
        self.input_src['batch_outputs'] = optools.wf_input
        self.input_src['from_outputs'] = optools.user_input
        self.input_src['operation'] = optools.wf_input
        self.input_src['var_name'] = optools.user_input
        self.input_type['from_outputs'] = optools.bool_type
        #self.input_type['operation'] = optools.str_type
        self.input_type['var_name'] = optools.str_type
        # defaults
        #self.inputs['from_inputs'] = True
        self.categories = ['MISC.SLACX OBJECT MANIPULATION']

    def run(self):
        batch_outputs = self.inputs['batch_outputs']
        operation = self.inputs['operation'].__name__
        var_name = self.inputs['var_name']
        num_items = len(batch_outputs)
        var_list = []
        for ii in range(num_items):
            if self.inputs['from_outputs'] == True:
                item = batch_outputs[ii][operation].outputs[var_name]
            else:
                item = batch_outputs[ii][operation].inputs[var_name]
            var_list.append(item)
        self.outputs['var_list'] = var_list
'''



class selectBatchItemFromOps(Operation):
    """Return a single output from batch."""

    def __init__(self):
        input_names = ['batch_outputs','index','operation','from_outputs','var_name']
        output_names = ['var']
        super(selectBatchItemFromOps, self).__init__(input_names, output_names)
        # source & type
        self.input_src['batch_outputs'] = optools.wf_input
        self.input_src['index'] = optools.user_input
        self.input_src['from_outputs'] = optools.user_input
        self.input_src['operation'] = optools.wf_input
        self.input_src['var_name'] = optools.user_input
        self.input_type['index'] = optools.int_type
        self.input_type['from_outputs'] = optools.bool_type
        self.input_type['operation'] = optools.str_type
        self.input_type['var_name'] = optools.str_type
        # defaults
        self.inputs['from_outputs'] = True
        self.categories = ['MISC.SLACX OBJECT MANIPULATION']

    def run(self):
        batch_outputs = self.inputs['batch_outputs']
        index = int(self.inputs['index'])
        op_name = self.inputs['operation']
        var_name = self.inputs['var_name']
        if self.inputs['from_outputs'] == True:
#            batch.outputs[index][]
            self.outputs['var'] = batch_outputs[index][op_name].outputs[var_name]
        else:
            self.outputs['var'] = batch_outputs[index][op_name].inputs[var_name]
