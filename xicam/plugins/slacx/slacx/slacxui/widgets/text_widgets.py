from ...slacxcore.operations import optools 
from ...slacxcore.operations.slacxop import Operation
from ...slacxcore.slacxtools import FileSystemIterator

unit_indent='&nbsp;&nbsp;&nbsp;&nbsp;'

def display_text(itm,indent=unit_indent):
    if type(itm).__name__ in ['str','unicode']:
        t = indent + '(str) <br>' + indent + '{}'.format(itm)
    elif isinstance(itm,dict):
        t = indent + '(dict)'
        for k,v in itm.items():
            t += '<br>' + indent + '{}: <br>{}'.format(k,display_text(v,indent+unit_indent))
    elif isinstance(itm,list):
        t = indent + '(list)'
        for i in range(len(itm)):
            t += '<br>' + indent + '{}: <br>{}'.format(i,display_text(itm[i],indent+unit_indent))
    elif isinstance(itm,Operation):
        t = indent + '(Operation)'
        t += '<br>' + indent + 'inputs: <br>{}'.format(display_text(itm.inputs,indent+unit_indent))
        t += '<br>' + indent + 'outputs: <br>{}'.format(display_text(itm.outputs,indent+unit_indent))
    elif isinstance(itm,optools.InputLocator):
        t = indent + '(InputLocator)'
        t += '<br>' + indent + 'src: {}'.format(optools.input_sources[itm.src])
        t += '<br>' + indent + 'type: {}'.format(optools.input_types[itm.tp])
        t += '<br>' + indent + 'val: {}'.format(itm.val)
        t += '<br>' + indent + 'data: {}'.format(itm.data)
    elif isinstance(itm,FileSystemIterator):
        t = indent + '(FileSystemIterator) - history'
        for p in itm.paths_done:
            t += '<br>' + indent + ' {}'.format(p)
    else:
        t = indent + '('+type(itm).__name__+')' + '<br>' + indent + '{}'.format(itm)
    return t
    

