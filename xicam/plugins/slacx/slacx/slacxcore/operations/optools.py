##### DEFINITIONS OF SOURCES FOR OPERATION INPUTS
input_sources = ['(select)','Text','Images','Operations','Filesystem','List builder','Tree builder'] 
text_input = 1
image_input = 2
op_input = 3
fs_input = 4
list_input = 5
tree_input = 6
valid_sources = [text_input,image_input,op_input,fs_input,list_input,tree_input]

##### VALID TYPES FOR TEXT BASED OPERATION INPUTS 
input_types = ['(select)','string','int','float','array','bool']
string_type = 1
int_type = 2
float_type = 3
array_type = 4
bool_type = 5
valid_types = [string_type,int_type,float_type,array_type,bool_type]

##### CONVENIENCE METHOD FOR PRINTING DOCUMENTATION
def parameter_doc(name,val,doc):
    return "name: {} \nvalue: {} \ndoc: {}".format(name,val,doc) 

##### CONVENIENCE CLASS FOR STORING OR LOCATING OPERATION INPUTS
class InputLocator(object):
    """
    The presence of an object of this type as input to an Operation 
    indicates that this input has not yet been loaded or computed.
    Objects of this class contain the information needed to find the relevant input data.
    If raw textual input is provided, it is stored in self.val after typecasting.
    """
    def __init__(self,src,val):
        if src < 0 or src > len(input_sources):
            msg = 'found input source {}, should be between 0 and {}'.format(
            src, len(input_sources))
            raise ValueError(msg)
        self.src = src
        self.val = val 

