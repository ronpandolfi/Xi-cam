import abc
import re
from collections import OrderedDict

import optools

class Operation(object):
    __metaclass__ = abc.ABCMeta
    """
    Abstract class template for implementing slacx operations.
    """

    def __init__(self,input_names,output_names):
        """
        The input_names and output_names (lists of strings)
        are used to specify names for the parameters 
        that will be used to perform the operation.
        These lists are used as keys to build dicts
        Operation.inputs and Operation.outputs.
        The Operation.categories attribute is a list that indicates
        where the operation can be found in the OpManager tree.
        The Operation will appear one time for each category in the list.
        Subcategories are indicated by a ".", for example:
        self.categories = ['CAT1','CAT2.SUBCAT','CAT3'].
        """
        self.inputs = OrderedDict()
        self.input_locator = OrderedDict() 
        self.outputs = OrderedDict() 
        self.input_doc = {}
        self.input_src = {}
        self.input_type = {}
        #self.output_container = {}
        self.output_doc = {}
        # For each of the var names, assign to None 
        for name in input_names: 
            self.input_src[name] = optools.no_input
            self.input_type[name] = optools.none_type
            self.input_locator[name] = None 
            self.inputs[name] = None
            self.input_doc[name] = None
        for name in output_names: 
            #self.output_container[name] = optools.OutputContainer() 
            self.outputs[name] = None
            self.output_doc[name] = None
        # Set default category to be 'MISC'
        self.categories = ['MISC']

    @abc.abstractmethod
    def run(self):
        """
        Operation.run() should use all of the items in Operation.inputs
        and set values for all of the items in Operation.outputs.
        """
        pass

    def description(self):
        """
        self.description() returns a string 
        documenting the input and output structure 
        and usage instructions for the Operation
        """
        return str(
        "Operation description: "
        + self.doc_as_string()
        + "\n\n--- Input ---"
        + self.input_description() 
        + "\n\n--- Output ---"
        + self.output_description())

    def doc_as_string(self):
        if self.__doc__:
            return re.sub("\s\s+"," ",self.__doc__.replace('\n','')) 
        else:
            return "no documentation found"

    def input_description(self):
        a = ""
        inp_indx = 0
        for name,val in self.inputs.items(): 
            if self.input_locator[name]:
                display_val = self.input_locator[name].val
            else:
                display_val = val 
            a = a + str("\n\nInput {}:\n".format(inp_indx) 
            + optools.parameter_doc(name,display_val,self.input_doc[name]))
            inp_indx += 1
        return a

    def output_description(self):
        a = ""
        out_indx = 0
        for name,val in self.outputs.items(): 
            a = a + str("\n\nOutput {}:\n".format(out_indx) 
            + optools.parameter_doc(name,val,self.output_doc[name]))
            out_indx += 1
        return a
                
    #def run_and_update(self):
    #    """
    #    Run the Operation and save its outputs in its output_locator 
    #    """
    #    self.run()
    #    self.save_outputs()

    #def save_outputs(self):
    #    """Loads the data from outputs[names] into output_container[names].data"""
    #    for name,d in self.outputs.items():
    #        self.output_container[name].data = d

class Realtime(Operation):
    __metaclass__ = abc.ABCMeta
    """
    Abstract class template for implementing real time execution operations.
    """
    def __init__(self,input_names,output_names):
        super(Realtime,self).__init__(input_names,output_names)

    @abc.abstractmethod
    def output_list(self):
        """
        Produce a list of OrderedDicts representing the outputs for each batch input.
        Each OrderedDict should be populated with [input_uri:input_value] pairs.
        """
        pass

    @abc.abstractmethod
    def input_iter(self):
        """
        Produce an iterator over OrderedDicts representing each set of inputs to run.
        Each dict should be populated with [input_uri:input_value] pairs.
        When there is no new set of inputs to run, should return None.
        """
        pass

    @abc.abstractmethod
    def input_routes(self):
        """
        Produce a list of [input_uri] routes 
        in the same order as the OrderedDicts 
        produced by Batch.input_iter()
        """
        pass

    def delay(self):
        """
        Return the number of MILLIseconds to pause between iterations.
        Overload this method to change the pause time- default is 1 second.
        """
        return 1000

    def saved_items(self):
        """
        Return a list of item uris to be saved after each execution.
        Default returns empty list, saves all ops.
        """
        return []

class Batch(Operation):
    __metaclass__ = abc.ABCMeta
    """
    Abstract class template for implementing batch execution operations.
    """
    def __init__(self,input_names,output_names):
        super(Batch,self).__init__(input_names,output_names)


    @abc.abstractmethod
    def output_list(self):
        """
        Produce a list of OrderedDicts representing the outputs for each batch input.
        Each OrderedDict should be populated with [input_uri:input_value] pairs.
        """
        pass

    @abc.abstractmethod
    def input_list(self):
        """
        Produce a list of OrderedDicts representing each set of inputs for the Batch to run.
        Each OrderedDict should be populated with [input_uri:input_value] pairs.
        """
        pass

    @abc.abstractmethod
    def input_routes(self):
        """
        Produce a list of the input routes used by the Batch,
        in the same order as each of the OrderedDicts 
        provided by Batch.input_list()
        """
        pass

    def saved_items(self):
        """
        Return a list of items to be saved after each execution.
        Default returns empty list, saves all Operations.
        """
        return []


#class Workflow(Operation):
#    """
#    A Workflow has the same interface as an Operation
#    but is in fact a tree (describing a graph) of Operations,
#    as implemented in the slacx workflow manager (slacxwfman.WfManager).
#    The Inputs to a Workflow are the set of all inputs
#    to the Operations in the graph that are in the "INPUT" category.
#    The run() method of an Workflow calls on the WfManager
#    to execute the Operations by whatever means.
#    """
#
#    def __init__(self,wfman,wfname,desc):
#        # TODO: Find input and output names from the wfman
#        super(Workflow,self).__init__(input_names,output_names)
#        self.categories = ['WORKFLOW']
#        self.wfname = wfname 
#        self.quick_desc = desc
#        self.wfman = wfman
#
#    def description(self):
#        """
#        self.description() returns a string 
#        documenting the input and output structure 
#        and usage instructions for the Workflow.
#        """
#        msg = "Workflow: "+self.quick_desc+"\n"
#        # TODO: Loop through the Operations tree
#        # Would be cool to just print this tree eventually. 
#        return msg
#
#    def run(self):
#        self.wfman.run_wf_serial()
#
#    def inputs_description(self):
#        msg = ""
#        # TODO: Loop through the Operations tree
#        return msg 
#
#    def outputs_description(self):
#        msg = ""
#        # TODO: Loop through the Operations tree
#        return msg 


