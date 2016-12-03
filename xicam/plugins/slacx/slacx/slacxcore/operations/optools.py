from PySide import QtCore

import slacxop

##### TODO: THIS, MORE ELEGANTLY
# definitions for operation input sources 
input_sources = ['None','User Input','Filesystem','Workflow','Batch'] 
no_input = 0
user_input = 1
fs_input = 2
wf_input = 3
batch_input = 4 
valid_sources = [no_input,user_input,fs_input,wf_input,batch_input]

# supported types for operation inputs
input_types = ['none','auto','string','integer','float','boolean','list']
none_type = 0
auto_type = 1
str_type = 2
int_type = 3
float_type = 4
bool_type = 5
list_type = 6
valid_types = [none_type,auto_type,str_type,int_type,float_type,bool_type,list_type]

# tags and indices for inputs and outputs trees
inputs_tag = 'inputs'
outputs_tag = 'outputs'
inputs_idx = 0
outputs_idx = 1

def cast_type_val(tp,val):
    if tp == none_type:
        val = None 
    elif tp == int_type:
        val = int(val)
    elif tp == float_type:
        val = float(val)
    elif tp == str_type:
        val = str(val)
    elif tp == bool_type:
        val = bool(val)
    elif tp == list_type:
        # val will be a list of things, already typecast by the list builder 
        val = list(val)
    else:
        msg = 'type selection {}, should be one of {}'.format(src,valid_types)
        raise ValueError(msg)
    return val

def get_child_dict(x):
    """Get a dict to be used in building TreeItem children from structured data object x"""
    if isinstance(x,slacxop.Operation):
        d = {} 
        d[inputs_tag] = x.input_locator 
        #d[str(outputs_idx)] = x.output_container
        d[outputs_tag] = x.outputs
    elif isinstance(x,dict):
        d = x 
    elif isinstance(x,list):
        d = {str(i):x[i] for i in range(len(x))} 
    else:
        d = {} 
    return d

def val_list(il):
    if il.tp == list_type:
        return il.val
    else:
        return [il.val]

def parse_wf_input(wfman,il,op):
    uris = val_list(il)
    #print uris
    for uri in uris:
        #print uri
        uri_parts = uri.split('.')
        if not len(uri_parts) < 3:
            io_type = uri_parts[1]
            if io_type == inputs_tag:
                #print 'seeking input {}'.format(uri)
                inprouteflag = False
                if isinstance(op,slacxop.Batch) or isinstance(op,slacxop.Realtime):
                    inprouteflag = uri in op.input_routes()
                if inprouteflag:
                    # a uri used to direct a batch executor in setting data.
                    # It should be returned directly- the batch will use it as is.
                    return uri 
                else:
                    # an input uri... trusting that this input has already been loaded,
                    # grab the data from the InputLocator at that uri and return it.
                    # TODO: make sure this is coherent with wfman.upstream_stack() 
                    itm, idx = wfman.get_from_uri(uri)
                    il = itm.data 
                    return il.data
            else:
                #print 'seeking output {}'.format(uri)
                itm, idx = wfman.get_from_uri(uri)
                #print 'found {}'.format(itm.data)
                return itm.data
        else:
            #print 'return tree item at {}'.format(uri)
            # return the whole tree item?
            itm, indx = wfman.get_from_uri(uri)
            return itm

        #downstreamflag = False
        #if isinstance(op,slacxop.Batch) or isinstance(op,slacxop.Realtime):
        #    downstreamflag = uri in op.downstream_ops()
        #if downstreamflag:
        #    # a uri used to locate downstream Operations.
        #    # the entire TreeItem containing the operation should be returned.
        #    itm,idx = wfman.get_from_uri(uri)
        #    return itm
        #else:
        #itm,idx = wfman.get_from_uri(uri)
        #return itm.data
    #elif len(uri_parts) == 2:
    #    # An entire inputs or outputs dict is requested.
    #    itm,idx = wfman.get_from_uri(uri)
    #    return itm.data
    #else:
        # A specific input or output is requested.
        #if io_type == outputs_tag:
            # uri points to an op output. 
            # Get the item from the uri.
            #itm, indx = wfman.get_from_uri(uri)
            #return itm.data
            # Unpackage the OutputContainer
            #oc = itm.data
            #return oc.data

#class OutputContainer(object):
#    """
#    Objects of this class are used as containers for outputs of an Operation.
#    OutputContainer.data should be None at least until the Operation runs,
#    at which point the WfManager should replace it with the actual output.
#    """
#    def __init__(self,data=None):
#        self.data = data 

class InputLocator(object):
    """
    Objects of this class are used as containers for inputs to an Operation,
    and should by design contain the information needed to find the relevant input data.
    After the data is loaded, it should be stored in InputLocator.data.
    """
    def __init__(self,src=no_input,tp=none_type,val=None):
        #if src not in valid_sources: 
        #    msg = 'found input source {}, should be one of {}'.format(src, valid_sources)
        self.src = src
        self.tp = tp
        self.val = val 
        self.data = None 

def parameter_doc(name,value,doc):
    #if type(value).__name__ == 'InputLocator':
    if isinstance(value, InputLocator):
        src_str = input_sources[value.src]
        tp_str = input_types[value.tp]
        d = value.data
        return "- name: {} \n- source: {} \n- type: {} \n- value: {} \n- doc: {}".format(name,src_str,tp_str,d,doc) 
    else:
        val_str = str(value)
        tp_str = type(value).__name__
        return "- name: {} \n- type: {} \n- value: {} \n- doc: {}".format(name,tp_str,val_str,doc) 
        
#def loader_extensions():
#    return str(
#    "ALL (*.*);;"
#    + "TIFF (*.tif *.tiff);;"
#    + "RAW (*.raw);;"
#    + "MAR (*.mar*)"
#    )

