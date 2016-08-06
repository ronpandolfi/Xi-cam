import os
import time
from collections import OrderedDict
from copy import deepcopy
from functools import partial, wraps
import types
import numpy as np
from PySide import QtGui, QtCore
from PySide.QtUiTools import QUiLoader
from pipeline import msg
import config
import fncwidgets
import reconpkg
import ui
import yamlmod
from xicam import threads

# imports for functions exposed in pipeline
import pipelinefunctions, dxchange

functions = []
recon_function = None
currentindex = 0
layout = None

# Center of Rotation correction functions (corrects for padding and upsampling/downsampling)
cor_offset = None
cor_scale = lambda x: x

def reset_cor():
    global cor_offset, cor_scale
    cor_offset = None
    cor_scale = lambda x: x


def static_var(varname, value):
    def decorate(func):
        setattr(func, varname, value)
        return func
    return decorate


def lock_function_params(boolean):
    global functions
    for func in functions:
        func.allReadOnly(boolean)

def set_pipeline_from_preview(pipeline, setdefaults=False):
    clear_functions()
    for func, subfuncs in pipeline.iteritems():
        for subfunc in subfuncs:
            funcWidget = add_function(func, subfunc)
            for param, value in subfuncs[subfunc].iteritems():
                if param == 'Package':
                    continue
                elif param == 'Input Functions':
                    for ipf, sipfs in value.iteritems():
                        ifwidget = funcWidget.addInputFunction(ipf, list(sipfs.keys())[0])
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

def update_function_parameters(funcs):
    for f in funcs:
        f.updateParamsDict()
        if f.input_functions is not None:
            update_function_parameters(funcs=f.input_functions)


@threads.method(callback_slot=lambda x: run_preview_recon(*x), lock=threads.mutex)
def pipeline_preview_action(widget, callback, finish_call=None, update=True, slc=None, fixed_funcs=None):
    global functions

    if len(functions) < 1:
        return None, None, None
    elif 'Reconstruction' not in [func.func_name for func in functions]:
        QtGui.QMessageBox.warning(None, 'Reconstruction method required',
                                  'You have to select a reconstruction method to run a preview')
        return None, None, None
    if fixed_funcs is None:
        fixed_funcs = {}
    return construct_preview_pipeline(widget, callback, update=update, slc=slc, fixed_funcs=fixed_funcs)


def set_center_correction(name, param_dict):
    global cor_offset, cor_scale
    if 'Padding' in name and param_dict['axis'] == 2:
        n = param_dict['npad']
        cor_offset = lambda x: cor_scale(x) + n
    elif 'Downsample' in name and param_dict['axis'] == 2:
        s = param_dict['level']
        cor_scale = lambda x: x / 2 ** s
    elif 'Upsample' in name and param_dict['axis'] == 2:
        s = param_dict['level']
        cor_scale = lambda x: x * 2 ** s


def correct_center(f):
    @wraps(f)
    def corrected_update(fpartial, fname, param_dict, *args, **kwargs):
        global cor_scale, cor_offset
        if fname in ('Padding', 'Downsample', 'Upsample'):
            set_center_correction(fname, param_dict)
        p = f(fpartial, fname, param_dict, *args, **kwargs)
        if 'Reconstruction' in fname:
            if cor_offset is None:
                cor_offset = cor_scale
            p.keywords['center'] = cor_offset(p.keywords['center'])
            reset_cor()
        return p
    return corrected_update


@correct_center
def update_function_partial(fpartial, fname, param_dict, fargs, datawidget, input_partials=None, slc=None, ncore=None):
    kwargs = fpartial.keywords
    for key in fargs:
        if key in 'flats':
            kwargs[key] = datawidget.getflats(slc=slc)
        if key in 'darks':
            kwargs[key] = datawidget.getdarks(slc=slc)
        if key in 'ncore' and ncore is not None:
            kwargs[key] = ncore
    if input_partials is not None:
        for pname, slices, ipartial in input_partials:
            if pname is None:
                continue
            pargs = []
            if slices is not None:
                map(pargs.append, (map(datawidget.data.fabimage.__getitem__, slices)))
            kwargs[pname] = ipartial(*pargs)
            if param_dict is not None and pname in param_dict.keys():
                param_dict[pname] = kwargs[pname]
    return partial(fpartial, *fpartial.args, **kwargs)


def construct_preview_pipeline(widget, callback, fixed_funcs=None, update=True, slc=None):
    global functions, cor_scale, recon_function
    if fixed_funcs is None:
        fixed_funcs = {}
    lock_function_params(True)  # you probably do not need this anymore but maybe you do...
    params = OrderedDict()
    funstack = []
    for func in functions:
        if not func.enabled:
            continue
        elif func.func_name == 'Write':
            params[func.func_name] = {func.subfunc_name: deepcopy(func.getParamDict(update=update))}
            continue
        # fixed_funcs used for parameter range tests to avoid updating the parameter based on the value in UI
        if func.subfunc_name in fixed_funcs:
            params[func.func_name] = {func.subfunc_name: fixed_funcs[func.subfunc_name][0]}
            fpartial = fixed_funcs[func.subfunc_name][1]
        else:
            params[func.func_name] = {func.subfunc_name: deepcopy(func.getParamDict(update=update))}
            fpartial = func.partial

        p = update_function_partial(fpartial, func.name, params[func.func_name][func.subfunc_name],
                                    func.args_complement, widget, input_partials=func.input_partials, slc=slc)

        funstack.append(p)
        if func.input_functions is not None:
            in_dict = {infunc.func_name: {infunc.subfunc_name: deepcopy(infunc.getParamDict(update=update))}
                                          for infunc in func.input_functions if infunc is not None
                                          and infunc.enabled}
            if in_dict:
                params[func.func_name][func.subfunc_name].update({'Input Functions': in_dict})
    lock_function_params(False)
    return funstack, widget.getsino(slc), partial(callback, params)


@QtCore.Slot(list, np.ndarray, types.FunctionType)
def run_preview_recon(funstack, initializer, callback):
    if funstack is not None:
        bg_reduce = threads.method(callback_slot=callback, lock=threads.mutex)(reduce)
        bg_reduce(lambda f1, f2: f2(f1), funstack, initializer)  # fold


def run_full_recon(widget, proj, sino, sino_p_chunk, ncore, update_call=None,
                   finish_call=None, interrupt_signal=None):
    global functions
    lock_function_params(True)
    funcs, params = [], OrderedDict()
    for f in functions:
        if not f.enabled and f.func_name != 'Reconstruction':
            continue
        params[f.subfunc_name] = deepcopy(f.getParamDict(update=update))
        funcs.append([deepcopy(f.partial), f.name, deepcopy(f.param_dict), f.args_complement,
                      deepcopy(f.input_partials)])
    lock_function_params(False)
    bg_recon_iter = threads.iterator(callback_slot=update_call, finished_slot=finish_call,
                                     interrupt_signal=interrupt_signal)(_recon_iter)
    bg_recon_iter(widget, funcs, proj, sino, sino_p_chunk, ncore)
    return params


def _recon_iter(datawidget, fpartials, proj, sino, sino_p_chunk, ncore):
    write_start = sino[0]
    nchunk = ((sino[1] - sino[0]) // sino[2] - 1) // sino_p_chunk + 1
    total_sino = (sino[1] - sino[0] - 1) // sino[2] + 1
    if total_sino < sino_p_chunk:
        sino_p_chunk = total_sino

    for i in range(nchunk):
        init = True
        start, end = i*sino[2]*sino_p_chunk + sino[0], (i + 1)*sino[2]*sino_p_chunk + sino[0]
        end = end if end < sino[1] else sino[1]

        for fpartial, fname, param_dict, fargs, ipartials in fpartials:
            ts = time.time()
            yield 'Running {0} on slices {1} to {2} from a total of {3} slices...'.format(fname, start,
                                                                                          end, total_sino)
            fpartial = update_function_partial(fpartial, fname, param_dict, fargs, datawidget,
                                               slc=(slice(*proj), slice(start, end, sino[2]), slice(None, None, None)),
                                               ncore=ncore, input_partials=ipartials)
            if init:
                tomo = datawidget.getsino(slc=(slice(*proj), slice(start, end, sino[2]),
                                                        slice(None, None, None)))
                init = False
            elif 'Tiff' in fname:
                fpartial.keywords['start'] = write_start
                write_start += tomo.shape[0]
            # elif 'Reconstruction' in fname:
            #     # Reset input_partials to None so that centers and angle vectors are not computed in every iteration
            #     # and set the reconstruction partial to the updated one.
            #     if ipartials is not None:
            #         ind = next((i for i, names in enumerate(fpartials) if fname in names), None)
            #         fpartials[ind][0], fpartials[ind][4] = fpartial, None
            #     tomo = fpartial(tomo)
            tomo = fpartial(tomo)
            yield ' Finished in {:.3f} s\n'.format(time.time() - ts)
