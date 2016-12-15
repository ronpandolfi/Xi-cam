from ..slacxop import Operation

class Identity(Operation):
    """An Operation testing class, loads its input into its output"""

    def __init__(self):
        input_names = ['data']
        output_names = ['data']
        super(Identity,self).__init__(input_names,output_names) 
        self.input_doc['data'] = 'this can actually be anything'
        self.output_doc['data'] = 'this ends up being whatever the input was'
        self.categories = ['TESTS'] 
        
    def run(self):
        self.outputs['data'] = self.inputs['data']
