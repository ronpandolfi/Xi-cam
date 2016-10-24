
from ..operations.slacxop import Operation

class Workflow(Operation):
    """
    A Workflow has the same interface as an Operation
    but is in fact a tree (describing a graph) of Operations,
    as implemented in the slacx workflow manager (slacxwfman.WfManager).
    The Inputs to a Workflow are the set of all inputs
    to the Operations in the graph that are in the "INPUT" category.
    The run() method of an Workflow calls on the WfManager
    to execute the Operations by whatever means.
    """

    def __init__(self,wfman,wfname,desc):
        # TODO: Find input and output names from the wfman
        super(Workflow,self).__init__(input_names,output_names)
        self.categories = ['WORKFLOW']
        self.wfname = wfname 
        self.quick_desc = desc
        self.wfman = wfman

    def description(self):
        """
        self.description() returns a string 
        documenting the input and output structure 
        and usage instructions for the Workflow.
        """
        msg = "Workflow: "+self.quick_desc+"\n"
        # TODO: Loop through the Operations tree
        # Would be cool to just print this tree eventually. 
        return msg

    def run(self):
        self.wfman.run_wf_serial()

    def inputs_description(self):
        msg = ""
        # TODO: Loop through the Operations tree
        return msg 

    def outputs_description(self):
        msg = ""
        # TODO: Loop through the Operations tree
        return msg 


