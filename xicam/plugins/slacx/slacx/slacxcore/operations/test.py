from slacxop import Operation

import optools

class TestLoadBool(Operation):

    def __init__(self):
        input_names = ['bool']
        output_names = ['bool']
        super(TestLoadBool,self).__init__(input_names,output_names) 
        self.input_src['bool'] = optools.user_input
        self.input_src['bool'] = optools.user_input
        self.input_type['bool'] = optools.bool_type
        self.inputs['bool'] = True
        self.categories = ['TESTS'] 
        
    def run(self):
        self.outputs['data'] = self.inputs['data']
