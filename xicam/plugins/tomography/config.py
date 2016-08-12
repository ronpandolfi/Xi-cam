
__author__ = "Luis Barroso-Luque"
__copyright__ = "Copyright 2016, CAMERA, LBL, ALS"
__credits__ = ["Ronald J Pandolfi", "Dinesh Kumar", "Singanallur Venkatakrishnan", "Luis Luque", "Alexander Hexemer"]
__license__ = ""
__version__ = "1.2.1"
__maintainer__ = "Ronald J Pandolfi"
__email__ = "ronpandolfi@lbl.gov"
__status__ = "Beta"


import os
from collections import OrderedDict
import yaml
from pipeline import msg
from xicam.modpkgs import yamlmod

PARAM_TYPES = {'int': int, 'float': float}

# Load yaml with names of all available functions in pipeline
with open('yaml/tomography/functions.yml','r') as stream:
    funcs=yaml.load(stream)

# Load parameter data for available functions
parameter_files = ('tomopy_function_parameters.yml',
                   'aux_function_parameters.yml',
                   'dataexchange_function_parameters.yml',
                   'astra_function_parameters.yml')
                   #'mbir_function_parameters.yml')
parameters = {}

for file in parameter_files:
    with open('yaml/tomography/'+file ,'r') as stream:
        parameters.update(yaml.load(stream))

# Load dictionary with pipeline names and function names
with open('yaml/tomography/function_names.yml','r') as stream:
    names=yaml.load(stream)

# Add reconstruction methods to function name dictionary, but include the package the method is in
for algorithm in funcs['Functions']['Reconstruction']['TomoPy']:
    names[algorithm] = ['recon', 'tomopy']

for algorithm in funcs['Functions']['Reconstruction']['Astra']:
    names[algorithm] = ['recon', 'astra']

# Load dictionary with function parameters to be retrieved from metadatas
with open('yaml/tomography/als832_function_defaults.yml','r') as stream:
    als832defaults = yaml.load(stream)


def load_pipeline(yaml_file):
    global functions, currentindex
    with open(yaml_file, 'r') as y:
        pipeline = yamlmod.ordered_load(y)
    return pipeline


def save_function_pipeline(pipeline, file_name):
    if file_name != '':
        file_name = file_name.split('.')[0] + '.yml'
        with open(file_name, 'w') as y:
            yamlmod.ordered_dump(pipeline, y)


def set_als832_defaults(mdata, funcwidget_list):
    for f in funcwidget_list:
        if f is None:
            continue
        if f.subfunc_name in als832defaults:
            for p in f.params.children():
                if p.name() in als832defaults[f.subfunc_name]:
                    try:
                        v = mdata[als832defaults[f.subfunc_name][p.name()]['name']]
                        t = PARAM_TYPES[als832defaults[f.subfunc_name][p.name()]['type']]
                        v = t(v) if t is not int else t(float(v))  # String literals for ints should not have 0's
                        if 'conversion' in als832defaults[f.subfunc_name][p.name()]:
                            v *= als832defaults[f.subfunc_name][p.name()]['conversion']
                        p.setDefault(v)
                        p.setValue(v)
                    except KeyError as e:
                        msg.logMessage('Key {} not found in metadata. Error: {}'.format(p.name(), e.message),
                                       level=40)
        elif f.func_name == 'Write':
            outname = os.path.join(os.path.expanduser('~'), *2*('RECON_' + mdata['dataset'],))
            f.params.child('fname').setValue(outname)
        if f.input_functions:
            set_als832_defaults(mdata, funcwidget_list=f.input_functions.values())


def extract_pipeline_dict(funwidget_list):
    d = OrderedDict()
    for f in funwidget_list:
        d[f.func_name] = {f.subfunc_name: {'Parameters': {p.name(): p.value() for p in f.params.children()}}}
        d[f.func_name][f.subfunc_name]['Enabled'] = f.enabled
        if f.func_name == 'Reconstruction':
            d[f.func_name][f.subfunc_name].update({'Package': f.packagename})
        for param, ipf in f.input_functions.iteritems():
            if 'Input Functions' not in d[f.func_name][f.subfunc_name]:
                d[f.func_name][f.subfunc_name]['Input Functions'] = {}
            id = {ipf.func_name: {ipf.subfunc_name: {'Parameters': {p.name(): p.value() for p in ipf.params.children()}}}}
            d[f.func_name][f.subfunc_name]['Input Functions'][param] = id
    return d