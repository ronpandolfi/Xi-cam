# Users and developers should remove all comments from this template.
# All text outside comments that is meant to be removed or replaced 
# is <written within angle brackets>.

# Operations implemented as python classes 
# have a common interface for communicating 
# with the slacx workflow manager.
# That common interface is ensured by inheriting it
# from an abstract class called 'Operation'.
from slacxop import Operation
import optools

# Name the operation, specify inheritance (Operation)
class <OperationName>(Operation):
    # Give a brief description of the operation
    # bracketed by """triple-double-quotes"""
    """<Description of Operation>"""

    # Write an __init__() function for the Operation.
    def __init__(self):
        # Name the input and output data/parameters for your operation.
        # Format names as 'single_quotes_without_spaces'.
        input_names = ['<input_name_1>','<input_name_2>',<...>]
        output_names = ['<output_name_1>','<output_name_2>',<...>]
        # Call the __init__ method of the Operation abstract (super)class.
        # This instantiates {key:value} dictionaries of inputs and outputs, 
        # which have keys generated from input_names and output_names.
        # All values in the dictionary are initialized as None. 
        super(<OperationName>,self).__init__(input_names,output_names)
        # Write a free-form documentation string describing each item
        # that was named in input_names and output_names.
        self.input_doc['<input_name_1>'] = '<expectations for input 1>'
        self.input_doc['<input_name_2>'] = '<etc>'
        self.output_doc['<output_name_1>'] = '<form of output 1>'
        self.output_doc['<output_name_2>'] = '<etc>'
        # Categorize the operation. Multiple categories are acceptable.
        # Indicate subcategories with a '.' character.
        self.categories = ['<CAT1>','<CAT2>.<SUBCAT1>',<...>]
        # OPTIONAL: set default sources and types for the operation inputs.
        # Default types are only used for textual inputs.
        # Valid sources: optools.fs_input (read input from filesystem), 
        #   optools.op_input (input from another operation's output), 
        #   optools.text_input (manual text input)
        self.input_src['<input_name_1>'] = <optools.some_source>
        self.input_src['<input_name_2>'] = <etc>
        # Valid types: optools.str_type (string), optools.int_type (integer),
        #   optools.float_type (floating-point number), optools.bool_type (boolean)
        self.input_type['<input_name_1>'] = <optools.some_type>
        self.input_type['<input_name_2>'] = <etc>
        
    # Write a run() function for this Operation.
    def run(self):
        # Optional- create references in the local namespace for cleaner code.
        <inp1> = self.inputs['<input_name_1>']
        <inp2> = self.inputs['<input_name_2>']
        <etc>
        # Perform the computation
        < ... >
        # Save the output
        self.outputs['<output_name_1>'] = <computed_value_1>
        self.outputs['<output_name_2>'] = <etc>

