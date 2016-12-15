from slacxop import Operation

import optools

class BuildList(Operation):
    """
    Builds a list from an input string 
    by splitting the string at ',' characters
    """

    def __init__(self):
        input_names = ['text']
        output_names = ['list']
        super(BuildList,self).__init__(input_names,output_names) 
        self.input_doc['text'] = 'comma-separated text to be packaged as an array'
        self.output_doc['list'] = 'output from array constructor'
        self.categories = ['INPUT.TESTS'] 
        self.input_src['text'] = optools.user_input
        self.input_type['text'] = optools.str_type
        self.inputs['text'] = '0,1,2,3'
        
    def run(self):
        inp_str = self.inputs['text']
        self.outputs['list'] = inp_str.replace(' ','').split(',')      


 
