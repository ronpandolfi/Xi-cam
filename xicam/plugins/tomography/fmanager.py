import os
import time
from collections import OrderedDict
from copy import deepcopy
from functools import partial, wraps
from PySide import QtGui, QtCore
from PySide.QtUiTools import QUiLoader
import fdata
import fwidgets
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

def clear_action():
    global functions
    if len(functions) == 0:
        return

    value = QtGui.QMessageBox.question(None, 'Delete functions',
                                       'Are you sure you want to clear ALL functions?',
                                       (QtGui.QMessageBox.Yes | QtGui.QMessageBox.Cancel))
    if value is QtGui.QMessageBox.Yes:
        clear_functions()


def clear_functions():
    global functions
    for feature in functions:
        feature.deleteLater()
        del feature
    functions = []
    ui.showform(ui.blankform)


def add_action(function, subfunction):
    global functions, recon_function
    if function in [func.func_name for func in functions]:
        value = QtGui.QMessageBox.question(None, 'Adding duplicate function',
                                           '{} function already in pipeline.\n'
                                           'Are you sure you need another one?'.format(function),
                                           (QtGui.QMessageBox.Yes | QtGui.QMessageBox.No))
        if value is QtGui.QMessageBox.No:
            return

    add_function(function, subfunction)


def add_function(function, subfunction):
    global functions, recon_function, currentindex
    try:
        package = reconpkg.packages[fdata.names[subfunction][1]]
    except KeyError:
        package = eval(fdata.names[subfunction][1])
    # if not hasattr(package, fdata.names[subfunction][0]):
    #     warnings.warn('{0} function not available in {1}'.format(subfunction, package))
    #     return

    currentindex = len(functions)
    if function == 'Reconstruction':
        if reconpkg.astra is not None and package == reconpkg.astra:
            func = fwidgets.AstraReconFuncWidget(function, subfunction, package)
        else:
            func = fwidgets.ReconFuncWidget(function, subfunction, package)
        recon_function = func
    else:
        func = fwidgets.FuncWidget(function, subfunction, package)
    functions.append(func)
    update()
    return func


def remove_function(index):
    global functions
    del functions[index]
    update()


def swap_functions(idx_1, idx_2):
    global functions, currentindex
    if idx_2 >= len(functions) or idx_2 < 0:
        return
    functions[idx_1], functions[idx_2] = functions[idx_2], functions[idx_1]
    currentindex = idx_2
    update()


def update():
    global layout, functions, recon_function
    assert isinstance(layout, QtGui.QVBoxLayout)

    for item in functions:
        layout.addWidget(item)
    widget = ui.centerwidget.currentWidget()
    if widget is not None:
        widget.wireupCenterSelection(recon_function)
        set_function_defaults(widget.data.header, functions)


def load_form(path):
    guiloader = QUiLoader()
    f = QtCore.QFile(path)
    f.open(QtCore.QFile.ReadOnly)
    form = guiloader.load(f)
    f.close()
    return form


def lock_function_params(boolean):
    global functions
    for func in functions:
        func.allReadOnly(boolean)


def load_function_pipeline(yaml_file, setdefaults=False):
    global functions, currentindex
    with open(yaml_file, 'r') as y:
        pipeline = yamlmod.ordered_load(y)
        set_function_pipeline(pipeline, setdefaults=setdefaults)


def set_function_pipeline(pipeline, setdefaults=False):
    clear_functions()
    # Way too many for loops, oops
    for func, subfuncs in pipeline.iteritems():
        for subfunc in subfuncs:
            funcWidget = add_function(func, subfunc)
            if 'Enabled' in subfuncs[subfunc] and not subfuncs[subfunc]['Enabled']:
                funcWidget.enabled = False
            if 'Parameters' in subfuncs[subfunc]:
                for param, value in subfuncs[subfunc]['Parameters'].iteritems():
                    child = funcWidget.params.child(param)
                    child.setValue(value)
                    if setdefaults:
                        child.setDefault(value)
            if 'Input Functions' in subfuncs[subfunc]:
                for ipf, sipfs in subfuncs[subfunc]['Input Functions'].iteritems():
                    for sipf in sipfs:
                        ifwidget = funcWidget.addInputFunction(ipf, sipf)
                        if 'Enabled' in sipfs[sipf] and not sipfs[sipf]['Enabled']:
                            ifwidget.enabled = False
                        if 'Parameters' in sipfs[sipf]:
                            for p, v in sipfs[sipf]['Parameters'].iteritems():
                                ifwidget.params.child(p).setValue(v)
                                if setdefaults: ifwidget.params.child(p).setDefault(v)



def create_pipeline_dict():
    d = OrderedDict()
    for f in functions:
        d[f.func_name] = {f.subfunc_name: {'Parameters': {p.name() : p.value() for p in f.params.children()}}}
        d[f.func_name][f.subfunc_name]['Enabled'] = f.enabled
        if f.func_name == 'Reconstruction':
            d[f.func_name][f.subfunc_name].update({'Package':f.packagename})
        if f.input_functions is not None:
            d[f.func_name][f.subfunc_name]['Input Functions'] = {}
            for ipf in f.input_functions:
                if ipf is not None:
                    id = {ipf.subfunc_name: {'Parameters': {p.name() : p.value() for p in ipf.params.children()}}}
                    d[f.func_name][f.subfunc_name]['Input Functions'][ipf.func_name] = id
    return d


def save_function_pipeline(pipeline):
    yaml_file = QtGui.QFileDialog.getSaveFileName(None, 'Save tomography pipeline file as', os.path.expanduser('~'),
                                                      '*.yml', selectedFilter='*.yml')[0]
    if yaml_file != '':
        yaml_file = yaml_file.split('.')[0] + '.yml'
        with open(yaml_file, 'w') as y:
            yamlmod.ordered_dump(pipeline, y)


def open_pipeline_file():
    pipeline_file = QtGui.QFileDialog.getOpenFileName(None, 'Open tomography pipeline file', os.path.expanduser('~'),
                                                      '*.yml')[0]
    if pipeline_file != '':
        load_function_pipeline(pipeline_file)


def set_function_defaults(mdata, funcs):
    for f in funcs:
        if f is None:
            continue
        if f.subfunc_name in fdata.als832defaults:
            for p in f.params.children():
                if p.name() in fdata.als832defaults[f.subfunc_name]:
                    v = mdata[fdata.als832defaults[f.subfunc_name][p.name()]['name']]
                    t = fdata.PARAM_TYPES[fdata.als832defaults[f.subfunc_name][p.name()]['type']]
                    v = t(v) if t is not int else t(float(v))  # String literals for ints should not have 0's
                    if 'conversion' in fdata.als832defaults[f.subfunc_name][p.name()]:
                        v *= fdata.als832defaults[f.subfunc_name][p.name()]['conversion']
                    p.setDefault(v)
                    p.setValue(v)
        elif f.func_name == 'Write':
            outname = os.path.join(os.path.expanduser('~'), *2*('RECON_' + mdata['dataset'],))
            f.params.child('fname').setValue(outname)

        if f.input_functions is not None:
            set_function_defaults(mdata, funcs=f.input_functions)


def update_function_parameters(funcs):
    for f in funcs:
        f.updateParamsDict()
        if f.input_functions is not None:
            update_function_parameters(funcs=f.input_functions)


def pipeline_preview_action(widget, callback, update=True, slc=None, fixed_funcs=None):
    global functions

    if len(functions) < 1:
        return None, None, None
    elif 'Reconstruction' not in [func.func_name for func in functions]:
        QtGui.QMessageBox.warning(None, 'Reconstruction method required',
                                  'You have to select a reconstruction method to run a preview')
        return None, None, None
    if fixed_funcs is None:
        fixed_funcs = {}

    construct_in_background = threads.method(construct_preview_pipeline,
                                             callback_slot=lambda x: run_preview_recon(*x),
                                             lock=threads.mutex)
    construct_in_background(widget, callback, update=update, slc=slc, fixed_funcs=fixed_funcs)


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


def run_preview_recon(funstack, initializer, callback):
    if funstack is not None:
        runnable = threads.RunnableMethod(reduce, method_args=((lambda f1, f2: f2(f1)), funstack, initializer),
                                          callback_slot=callback, lock=threads.mutex)
        threads.add_to_queue(runnable)


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
    runnable_it = threads.RunnableIterator(_recon_iter,
                                           generator_args=(widget, funcs, proj, sino, sino_p_chunk, ncore,),
                                           callback_slot=update_call, finished_slot=finish_call,
                                           interrupt_signal=interrupt_signal)
    threads.add_to_queue(runnable_it)
    return params


def _recon_iter(datawidget, fpartials, proj, sino, sino_p_chunk, ncore):
    write_start = sino[0]
    nchunk = ((sino[1] - sino[0]) // sino[2] - 1) // sino_p_chunk + 1
    total_sino = (sino[1] - sino[0]) // sino[2]
    if total_sino < sino_p_chunk:
        sino_p_chunk = total_sino

    for i in range(nchunk):
        init = True
        start, end = i * sino_p_chunk + sino[0], (i + 1) * sino_p_chunk + sino[0]
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
            # This needs to be fixed so that the center is only detected once &not re-detected in each chunk
            # elif 'Reconstruction' in fname:
            #     # Reset input_partials to None so that centers and angle vectors are not computed in every iteration
            #     # and set the reconstruction partial to the updated one.
            #     if ipartials is not None:
            #         ind = next((i for i, names in enumerate(fpartials) if fname in names), None)
            #         fpartials[ind][0], fpartials[ind][4] = fpartial, None
            #     tomo = fpartial(tomo)
            tomo = fpartial(tomo)
            yield ' Finished in {:.3f} s\n'.format(time.time() - ts)


def get_output_path():
    global functions
    write_funcs = [f for f in functions if f.func_name == 'Write']
    return write_funcs[-1].params.child('fname').value()
