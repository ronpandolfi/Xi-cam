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
        q_min = 0.02
        q_max = 0.6
        good_qvals = ((qvals > q_min) & (qvals < q_max))
        q_I_array = np.zeros((good_qvals.sum(),2))
        q_I_array[:,0] = qvals[good_qvals]
        q_I_array[:,1] = ivals[good_qvals]
        self.outputs['q_I_array'] = q_I_array



class Window_q_I_2(Operation):
    """
    From input iterables of *q_list_in* and *I_list_in*,
    produce two 1d vectors
    where q is greater than *q_min* and less than *q_min*
    """

    def __init__(self):
        input_names = ['q_list_in','I_list_in','q_min','q_max']
        output_names = ['q_list_out','I_list_out']
        super(Window_q_I_2,self).__init__(input_names,output_names)
        # docstrings
        self.input_doc['q_list_in'] = '1d iterable listing q values'
        self.input_doc['I_list_in'] = '1d iterable listing I values'
        self.input_doc['q_min'] = 'lowest value of q of interest'
        self.input_doc['q_max'] = 'highest value of q of interest'
        self.output_doc['q_list_out'] = '1d iterable listing q values where q is between *q_min* and *q_max*'
        self.output_doc['I_list_out'] = '1d iterable listing I values where q is between *q_min* and *q_max*'
        # source & type
        self.input_src['q_list_in'] = optools.wf_input
        self.input_src['I_list_in'] = optools.wf_input
        self.input_src['q_min'] = optools.user_input
        self.input_src['q_max'] = optools.user_input
        self.input_type['q_min'] = optools.float_type
        self.input_type['q_max'] = optools.float_type
        self.inputs['q_min'] = 0.02
        self.inputs['q_max'] = 0.6
        self.categories = ['PACKAGING']

    def run(self):
        qvals = self.inputs['q_list_in']
        ivals = self.inputs['I_list_in']
        q_min = self.inputs['q_min']
        q_max = self.inputs['q_max']
        if (q_min >= q_max):
            raise ValueError("*q_max* must be greater than *q_min*.")
        good_qvals = ((qvals > q_min) & (qvals < q_max))
        self.outputs['q_list_out'] = qvals[good_qvals]
        self.outputs['I_list_out'] = ivals[good_qvals]


