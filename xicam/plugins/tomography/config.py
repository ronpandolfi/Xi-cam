import os
from collections import OrderedDict
import yaml
import yamlmod
from pipeline import msg
import reconpkg

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


def load_pipeline(yaml_file, manager, setdefaults=False):
    global functions, currentindex
    with open(yaml_file, 'r') as y:
        pipeline = yamlmod.ordered_load(y)
        set_pipeline_from_yaml(manager, pipeline, setdefaults=setdefaults)


def set_pipeline_from_yaml(manager, pipeline, setdefaults=False):
    manager.removeAllFeatures()
    # Way too many for loops, oops... may want to restructure the yaml files
    for func, subfuncs in pipeline.iteritems():
        for subfunc in subfuncs:
            funcWidget = manager.addFunction(func, subfunc, package=reconpkg.packages[names[subfunc][1]])
            if 'Enabled' in subfuncs[subfunc] and not subfuncs[subfunc]['Enabled']:
                funcWidget.enabled = False
            if 'Parameters' in subfuncs[subfunc]:
                for param, value in subfuncs[subfunc]['Parameters'].iteritems():
                    child = funcWidget.params.child(param)
                    child.setValue(value)
                    if setdefaults:
                        child.setDefault(value)
            if 'Input Functions' in subfuncs[subfunc]:
                for param, ipfs in subfuncs[subfunc]['Input Functions'].iteritems():
                    for ipf, sipfs in ipfs.iteritems():
                        for sipf in sipfs:
                            if param in funcWidget.input_functions:
                                ifwidget = funcWidget.input_functions[param]
                            else:
                                ifwidget = manager.addInputFunction(funcWidget, param, ipf, sipf,
                                                                package=reconpkg.packages[names[sipf][1]])
                            if 'Enabled' in sipfs[sipf] and not sipfs[sipf]['Enabled']:
                                ifwidget.enabled = False
                            if 'Parameters' in sipfs[sipf]:
                                for p, v in sipfs[sipf]['Parameters'].iteritems():
                                    ifwidget.params.child(p).setValue(v)
                                    if setdefaults:
                                        ifwidget.params.child(p).setDefault(v)
                            ifwidget.updateParamsDict()
            funcWidget.updateParamsDict()


def set_pipeline_from_preview(manager, pipeline, setdefaults=False):
    manager.removeAllFeatures()
    for func, subfuncs in pipeline.iteritems():
        for subfunc in subfuncs:
            funcWidget = manager.addFunction(func, subfunc)
            for param, value in subfuncs[subfunc].iteritems():
                if param == 'Package':
                    continue
                elif param == 'Input Functions':
                    for ipf, sipfs in value.iteritems():
                        ifwidget = manager.addInputFunction(ipf, list(sipfs.keys())[0])
                        [ifwidget.params.child(p).setValue(v) for p, v in sipfs[sipfs.keys()[0]].items()]
                        if setdefaults:
                            [ifwidget.params.child(p).setDefault(v) for p, v in sipfs[sipfs.keys()[0]].items()]
                        ifwidget.updateParamsDict()
                else:
                    child = funcWidget.params.child(param)
                    child.setValue(value)
                    if setdefaults:
                        child.setDefault(value)
                funcWidget.updateParamsDict()


def save_function_pipeline(pipeline, file_name):
    if file_name != '':
        file_name = file_name.split('.')[0] + '.yml'
        with open(file_name, 'w') as y:
            yamlmod.ordered_dump(pipeline, y)


def set_als832_defaults(mdata, funcs):
    for f in funcs:
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
            set_als832_defaults(mdata, funcs=f.input_functions.values())


def extract_pipeline_dict(funwidget_list):
    d = OrderedDict()
    for f in funwidget_list:
        d[f.func_name] = {f.subfunc_name: {'Parameters': {p.name(): p.value() for p in f.params.children()}}}
        d[f.func_name][f.subfunc_name]['Enabled'] = f.enabled
        if f.func_name == 'Reconstruction':
            d[f.func_name][f.subfunc_name].update({'Package': f.packagename})
        if f.input_functions is not None:
            d[f.func_name][f.subfunc_name]['Input Functions'] = {}
            for ipf in f.input_functions:
                if ipf is not None:
                    id = {ipf.subfunc_name: {'Parameters': {p.name(): p.value() for p in ipf.params.children()}}}
                    d[f.func_name][f.subfunc_name]['Input Functions'][ipf.func_name] = id
    return d