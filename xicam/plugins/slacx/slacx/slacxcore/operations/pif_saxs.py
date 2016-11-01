from pypif import pif
import pypif.obj as pifobj

from slacxop import Operation
import optools

class SaxsToPifProperty(Operation):
    """
    Take a SAXS spectrum 
    produced from a nanoparticle solution synthesis experiment 
    and package it into a PIF.Property object.
    """

    def __init__(self):
        input_names = ['saxs_spectrum','saxs_metadata']
        output_names = ['pif_property']
        super(SaxsToPifProperty,self).__init__(input_names,output_names)
        self.input_doc['saxs_spectrum'] = str('n-by-2 array representing a calibrated, integrated, background-corrected saxs spectrum, '
                                            + 'with q in column 0 and intensity in column 1.')
        self.input_doc['saxs_metadata'] = str('dict containing data about the saxs spectrum')
        self.output_doc['pif_property'] = str('A PIF.Property object that will be built around the SAXS spectrum')
        self.categories = ['PACKAGING.PIF']
        self.input_src['saxs_spectrum'] = optools.op_input
        self.input_src['saxs_metadata'] = optools.op_input 
        
    # Write a run() function for this Operation.
    def run(self):
        saxspec = self.inputs['saxs_spectrum']
        saxsmd = self.inputs['saxs_metadata']
        # Perform the computation
        pif_prop = pifobj.Property()
        # Save the output
        self.outputs['pif_property'] = pif_prop

