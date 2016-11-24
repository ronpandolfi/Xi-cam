import numpy as np

from slacxop import Operation
import optools

class Window_q_I(Operation):
    """
    From input iterables of q and I, 
    produce an n-by-2 vector 
    where q is greater than 0.02 and less than 0.6
    """

    def __init__(self):
        input_names = ['q_list','I_list']
        output_names = ['q_I_array']
        super(Window_q_I,self).__init__(input_names,output_names)        
        self.input_src['q_list'] = optools.wf_input
        self.input_src['I_list'] = optools.wf_input
        self.input_doc['q_list'] = '1d iterable listing q values'
        self.input_doc['I_list'] = '1d iterable listing intensity values'
        self.output_doc['q_I_array'] = 'n-by-2 array with q, I pairs for q>0.02 and <0.6'
        self.categories = ['PACKAGING']

    def run(self):
        qvals = self.inputs['q_list']
        ivals = self.inputs['I_list']
        qi_out = []
        for i in range(len(qvals)):
            qval = qvals[i]
            ival = ivals[i]
            if qval > 0.02 and qval < 0.6:
                qi_out.append([qval,ival])
        self.outputs['q_I_array'] = np.array(qi_out)



