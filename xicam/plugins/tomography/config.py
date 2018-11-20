__author__ = "Luis Barroso-Luque, Holden Parks"
__copyright__ = "Copyright 2016, CAMERA, LBL, ALS"
__credits__ = ["Ronald J Pandolfi", "Dinesh Kumar", "Singanallur Venkatakrishnan", "Luis Luque",
               "Holden Parks", "Alexander Hexemer"]
__license__ = ""
__version__ = "1.2.1"
__maintainer__ = "Ronald J Pandolfi"
__email__ = "ronpandolfi@lbl.gov"
__status__ = "Beta"


import os
import numpy as np
import inspect
from collections import OrderedDict
import yaml
from pipeline import msg
from modpkgs import yamlmod

PARAM_TYPES = {'int': int, 'float': float}

# Load yaml with names of all available functions in pipeline
with open('xicam/yaml/tomography/functions.yml', 'r') as stream:
    funcs=yaml.load(stream)

# load various function dictionaries from function_info.yml file
parameters = {};
als832defaults = {};
aps_defaults = {};
names = {};
function_defaults = {}
with open('xicam/yaml/tomography/functions_info.yml', 'r') as stream:
    info = yaml.load(stream)
    for key in info.keys():
        # load parameter data for available functions
        if 'parameters' in info[key].keys():
            parameters[key] = info[key]['parameters']

        # load dictionary with function parameters to be retrieved from metadata
        try:
            als832defaults[key] = info[key]['conversions']['als']
        except KeyError:
            pass
        try:
            aps_defaults[key] = info[key]['conversions']['aps']
        except KeyError:
            pass

        # load dictionary with pipeline names and function names
        if 'name' in info[key].keys():
            names[key] = info[key]['name']

        # load dictionary of set defaults
        if 'defaults' in info[key].keys():
            function_defaults[key] = info[key]['defaults']

# Add reconstruction methods to function name dictionary, but include the package the method is in
for algorithm in funcs['Functions']['Reconstruction']['TomoPy']:
    names[algorithm] = ['recon', 'tomopy']

for algorithm in funcs['Functions']['Reconstruction']['Astra']:
    names[algorithm] = ['recon', 'astra']

for algorithm in funcs['Functions']['Reconstruction']['TomoCam']:
    names[algorithm] = ['recon','mbir']


def load_pipeline(yaml_file):
    """
    Load a workflow pipeline from a yaml file
    """

    with open(yaml_file, 'r') as y:
        pipeline = yamlmod.ordered_load(y)
    return pipeline


def save_function_pipeline(pipeline, file_name):
    """
    Save a workflow pipeline from dict

    Parameters
    ----------
    pipeline : dict
        dictionary specifying the workflow pipeline
    file_name : str
        file name to save as yml
    """

    if file_name != '':
        file_name = file_name.split('.')[0] + '.yml'
        with open(file_name, 'w') as y:
            yamlmod.ordered_dump(pipeline, y)


def set_als832_defaults(mdata, funcwidget_list, path, shape):
    """
    Set defaults for ALS Beamline 8.3.2 from dataset metadata

    Parameters
    ----------
    mdata : dict
        dataset metadata
    funcwidget_list : list of FunctionWidgets
        list of FunctionWidgets exposed in the UI workflow pipeline
    path: str
        path to dataset
    shape: tuple
        tuple containing dataset shape
    """
    from psutil import cpu_count
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
        elif f.func_name == 'Reader': #dataset specific read values
            set_reader_defaults(f, shape, cpu_count())

        elif f.func_name == 'Write': #dataset specific write values
            data_folders = {'bl832data-raw':'bl832data-scratch', 'data-raw':'data-scratch'}
            file_name = path.split("/")[-1].split(".")[0]
            working_dir = path.split(file_name)[0]
            for key in data_folders.keys():
                if key in working_dir:
                    user = working_dir.split('/' + key)[-1].split('/')[1]
                    mount = working_dir.split(key)[0]
                    working_dir = os.path.join(mount, data_folders[key], user)
            outname = os.path.join(working_dir, *2*('RECON_' + file_name,))
            f.params.child('parent folder').setValue(working_dir)
            f.params.child('parent folder').setDefault(working_dir)
            f.params.child('folder name').setValue('RECON_' + file_name)
            f.params.child('folder name').setDefault('RECON_' + file_name)
            f.params.child('file name').setValue('RECON_' + file_name)
            f.params.child('file name').setDefault('RECON_' + file_name)
            f.params.child('fname').setValue(outname)
            f.params.child('fname').setDefault(outname)
        if f.input_functions:
            set_als832_defaults(mdata, funcwidget_list=f.input_functions.values(), path=path, shape=shape)

def set_aps_defaults(mdata, funcwidget_list, path, shape):
    """
    Set defaults for ALS Beamline 8.3.2 from dataset metadata

    Parameters
    ----------
    mdata : dict
        dataset metadata
    funcwidget_list : list of FunctionWidgets
        list of FunctionWidgets exposed in the UI workflow pipeline
    path: str
        path to dataset
    shape: tuple
        tuple containing dataset shape
    """
    from psutil import cpu_count
    for f in funcwidget_list:
        if f is None:
            continue
        if aps_defaults and f.subfunc_name in aps_defaults:
            for p in f.params.children():
                if p.name() in aps_defaults[f.subfunc_name]:
                    try:
                        v = mdata[aps_defaults[f.subfunc_name][p.name()]['name']]
                        t = PARAM_TYPES[aps_defaults[f.subfunc_name][p.name()]['type']]
                        v = t(v) if t is not int else t(
                            float(v))  # String literals for ints should not have 0's
                        if 'conversion' in aps_defaults[f.subfunc_name][p.name()]:
                            v *= aps_defaults[f.subfunc_name][p.name()]['conversion']
                        p.setDefault(v)
                        p.setValue(v)
                    except KeyError as e:
                        msg.logMessage('Key {} not found in metadata. Error: {}'.format(p.name(), e.message),
                                       level=40)
        elif f.func_name == 'Reader':  # dataset specific read values
            set_reader_defaults(f, shape, cpu_count())

        elif f.func_name == 'Padding':
            pad = int(np.ceil((shape[2] * np.sqrt(2) - shape[2]) / 2))
            f.params.child('npad').setValue(pad)
            f.params.child('npad').setDefault(pad)
        elif f.func_name == 'Crop':
            pad = int(np.ceil((shape[2] * np.sqrt(2) - shape[2]) / 2))
            f.params.child('p11').setValue(pad)
            f.params.child('p11').setDefault(pad)
            f.params.child('p12').setValue(pad)
            f.params.child('p12').setDefault(pad)
            f.params.child('p21').setValue(pad)
            f.params.child('p21').setDefault(pad)
            f.params.child('p22').setValue(pad)
            f.params.child('p22').setDefault(pad)

        elif f.func_name == 'Write':  # dataset specific write values
            data_folders = {'bl832data-raw': 'bl832data-scratch', 'data-raw': 'data-scratch'}
            file_name = path.split("/")[-1].split(".")[0]
            working_dir = path.split(file_name)[0]
            for key in data_folders.keys():
                if key in working_dir:
                    user = working_dir.split('/' + key)[-1].split('/')[1]
                    mount = working_dir.split(key)[0]
                    working_dir = os.path.join(mount, data_folders[key], user)
            outname = os.path.join(working_dir, *2 * ('RECON_' + file_name,))
            f.params.child('parent folder').setValue(working_dir)
            f.params.child('parent folder').setDefault(working_dir)
            f.params.child('folder name').setValue('RECON_' + file_name)
            f.params.child('folder name').setDefault('RECON_' + file_name)
            f.params.child('file name').setValue('RECON_' + file_name)
            f.params.child('file name').setDefault('RECON_' + file_name)
            f.params.child('fname').setValue(outname)
            f.params.child('fname').setDefault(outname)
        if f.input_functions:
            set_als832_defaults(mdata, funcwidget_list=f.input_functions.values(), path=path, shape=shape)


def set_reader_defaults(reader_widget, shape, cpu):
    """
    Sets defaults for reader widget based on dataset size
    """
    reader_widget.params.child('start_sinogram').setLimits([0, shape[2]])
    reader_widget.params.child('end_sinogram').setLimits([0, shape[2]])
    reader_widget.params.child('step_sinogram').setLimits([1, shape[2] + 1])
    reader_widget.params.child('start_projection').setLimits([0, shape[0]])
    reader_widget.params.child('end_projection').setLimits([0, shape[0]])
    reader_widget.params.child('step_projection').setLimits([1, shape[0] + 1])
    reader_widget.params.child('start_width').setLimits([0, shape[1]])
    reader_widget.params.child('end_width').setLimits([0, shape[1]])
    reader_widget.params.child('step_width').setLimits([1, shape[2] + 1])

    reader_widget.params.child('end_sinogram').setValue(shape[2])
    reader_widget.params.child('end_sinogram').setDefault(shape[2])
    reader_widget.params.child('end_projection').setValue(shape[0])
    reader_widget.params.child('end_projection').setDefault(shape[0])
    reader_widget.params.child('end_width').setValue(shape[1])
    reader_widget.params.child('end_width').setDefault(shape[1])
    reader_widget.params.child('sinograms_per_chunk').setValue(cpu * 5)
    reader_widget.params.child('projections_per_chunk').setValue(cpu * 5)
    reader_widget.params.child('sinograms_per_chunk').setDefault(cpu * 5)
    reader_widget.params.child('projections_per_chunk').setDefault(cpu * 5)



def extract_pipeline_dict(funwidget_list):
    """
    Extract a dictionary from a FunctionWidget list in the appropriate format to save as a yml file

    Parameters
    ----------
    funwidget_list : list of FunctionWidgets
        list of FunctionWidgets exposed in the UI workflow pipeline

    Returns
    -------
    dict
        dictionary specifying the workflow pipeline
    """

    # list of parameter name exceptions, for the "Write" function
    ex_lst = ['file name', 'fname', 'folder name', 'parent folder']

    d = OrderedDict()
    count = 1
    for f in funwidget_list:
        # a bunch of special cases for the write function
        func_name = str(count) + ". " + f.func_name
        if "Write" in f.func_name:
            write_dict = OrderedDict()
            d[func_name] = OrderedDict({f.subfunc_name: write_dict})
            for child in f.params.children():
                if child.name() in ex_lst:
                    d[func_name][f.subfunc_name][child.name()] = str(child.value())
                elif child.name() == "Browse":
                    pass
                else:
                    d[func_name][f.subfunc_name][child.name()] = child.value()
        else:
            d[func_name] = {f.subfunc_name: {'Parameters': {p.name(): p.value() for p in f.params.children()}}}
        d[func_name][f.subfunc_name]['Enabled'] = f.enabled
        if f.func_name == 'Reconstruction':
            d[func_name][f.subfunc_name].update({'Package': f.packagename})
        for param, ipf in f.input_functions.iteritems():
            if 'Input Functions' not in d[func_name][f.subfunc_name]:
                d[func_name][f.subfunc_name]['Input Functions'] = {}
            id = {ipf.func_name: {ipf.subfunc_name: {'Parameters': {p.name(): p.value() for p in ipf.params.children()}}}}
            d[func_name][f.subfunc_name]['Input Functions'][param] = id
        count += 1
    return d

def extract_runnable_dict(funwidget_list):
    """
    Extract a dictionary from a FunctionWidget list in the appropriate format to save as a python runnable.

    Parameters
    ----------
    funwidget_list : list of FunctionWidgets
        list of FunctionWidgets exposed in the UI workflow pipeline

    Returns
    -------
    dict
        dictionary specifying the workflow pipeline and important parameters
    """
    center_functions = {'find_center_pc': {'proj1': 'tomo[0]', 'proj2': 'tomo[-1]'},
                        'find_center': {'tomo': 'tomo', 'theta': 'theta'}, 'find_center_vo': {'tomo': 'tomo'}}

    d = OrderedDict()
    func_dict = OrderedDict(); subfuncs = OrderedDict()
    count = 1
    for f in funwidget_list:
        keywords = {}
        if not f.enabled or 'Reader' in f.name:
            continue

        func = "{}.{}".format(f.package, f._function.func_name)
        if 'xicam' in func:
            func = func.split(".")[-1]
        fpartial = f.partial
        for key, val in fpartial.keywords.iteritems():
            keywords[key] = val
        for arg in inspect.getargspec(f._function)[0]:
            if arg not in f.partial.keywords.iterkeys() or 'center' in arg:
                keywords[arg] = arg


        # get rid of degenerate keyword arguments
        if 'arr' in keywords and 'tomo' in keywords:
            keywords['tomo'] = keywords['arr']
            keywords.pop('arr', None)

        # special cases for the 'write' function
        if 'start' in keywords:
            keywords['start'] = 'start'
        if 'Write' in f.name:
            keywords.pop('parent folder', None)
            keywords.pop('folder name', None)
            keywords.pop('file name', None)


        if 'Reconstruction' in f.name:
            for param, ipf in f.input_functions.iteritems():
                if 'theta' in param or 'center' in param:
                    subfunc = "{}.{}(".format(ipf.package,ipf._function.func_name)
                    for key, val in ipf.partial.keywords.iteritems():
                        subfunc += "{}={},".format(key, val) if not isinstance(val, str) \
                            else '{}=\'{}\','.format(key, val)
                    for cor_func in center_functions.iterkeys():
                        if ipf._function.func_name == cor_func:
                            for k, v in center_functions[cor_func].iteritems():
                                subfunc += "{}={},".format(k, v)
                    subfunc += ")"
                    subfuncs[param] = subfunc
            if 'astra' in keywords['algorithm']:
                keywords['algorithm'] = 'tomopy.astra'

        func_dict[str(count) + ". " + func] = keywords
        count += 1

    d['func'] = func_dict
    d['subfunc'] = subfuncs
    return d

