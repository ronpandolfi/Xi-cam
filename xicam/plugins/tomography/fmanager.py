from PySide.QtUiTools import QUiLoader
from PySide import QtGui, QtCore
import os
from collections import OrderedDict
from functools import partial
from copy import deepcopy
import yamlmod
from xicam import threads
import ui
import fwidgets
import reconpkg
import fdata
import warnings

FUNCTIONS_W_METADATA_DEFAULTS = ['Projection Angles', 'Phase Retrieval', 'Polar Mean Filter']
PARAM_TYPES = {'int': int, 'float': float}


functions = []
recon_function = None
currentindex = 0
layout = None

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


def add_action(function, subfunction, package=reconpkg.packages['tomopy']):
    global functions, recon_function
    if not hasattr(package, fdata.names[subfunction]): return
    if function in [func.func_name for func in functions]:
        value = QtGui.QMessageBox.question(None, 'Adding duplicate function',
                                           '{} function already in pipeline.\n'
                                           'Are you sure you need another one?'.format(function),
                                           (QtGui.QMessageBox.Yes | QtGui.QMessageBox.No))
        if value is QtGui.QMessageBox.No:
            return

    add_function(function, subfunction, package=package)


def add_function(function, subfunction, package=reconpkg.tomopy):
    global functions, recon_function, currentindex
    currentindex = len(functions)
    if function == 'Reconstruction':
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
    global layout, functions
    assert isinstance(layout, QtGui.QVBoxLayout)

    for i in range(layout.count()):
        layout.itemAt(i).parent = None

    for item in functions:
        layout.addWidget(item)

    w = ui.centerwidget.currentWidget()
    if w:
        set_function_defaults(w.widget.data.header)

def load_form(path):
    guiloader = QUiLoader()
    f = QtCore.QFile(path)
    f.open(QtCore.QFile.ReadOnly)
    form = guiloader.load(f)
    f.close()
    return form


def load():
    global functions, layout
    layout.setAlignment(QtCore.Qt.AlignBottom)


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
    for func, subfuncs in pipeline.iteritems():
        for subfunc in subfuncs:
            try:
                if func == 'Reconstruction':
                    try:
                        funcWidget = add_function(func, subfunc,
                                                  package=reconpkg.packages[subfuncs[subfunc]['Package']])
                    except KeyError:
                        funcWidget = add_function(func, subfunc)
                else:
                    funcWidget = add_function(func, subfunc)
                for param, value in subfuncs[subfunc].iteritems():
                    if param == 'Package':
                        continue
                    elif param == 'Input Functions':
                        for ipf, sipfs in value.iteritems():
                            ifwidget = funcWidget.addInputFunction(ipf, list(sipfs.keys())[0])
                            [ifwidget.params.child(p).setValue(v) for p, v in sipfs[sipfs.keys()[0]].items()]
                    else:
                        child = funcWidget.params.child(param)
                        child.setValue(value)
                        if setdefaults: child.setDefault(value)
            except (IndexError, AttributeError):
                #raise
                # TODO: make this failure more graceful
                warnings.warn('Failed to load subfunction: ' + subfunc)


def create_pipeline_dict():
    d = OrderedDict()
    for f in functions:
        d[f.func_name] = {f.subfunc_name: {p.name() : p.value() for p in f.params.children()}}
        if f.func_name == 'Reconstruction':
            d[f.func_name][f.subfunc_name].update({'Package':f.packagename})
        if f.input_functions is not None:
            d[f.func_name][f.subfunc_name]['Input Functions'] = {}
            for ipf in f.input_functions:
                if ipf is not None:
                    id = {ipf.subfunc_name: {p.name() : p.value() for p in ipf.params.children()}}
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


def set_function_defaults(mdata, funcs=functions):
    global FUNCTIONS_W_METADATA_DEFAULTS, PARAM_TYPES
    for f in funcs:
        if f.subfunc_name in FUNCTIONS_W_METADATA_DEFAULTS:
            for p in f.params.children():
                if p.name() in fdata.als832defaults[f.func_name]:
                    v = mdata[fdata.als832defaults[f.func_name][p.name()]['name']]
                    v = PARAM_TYPES[fdata.als832defaults[f.func_name][p.name()]['type']](v)
                    if 'conversion' in fdata.als832defaults[f.func_name][p.name()]:
                        v *= fdata.als832defaults[f.func_name][p.name()]['conversion']
                    p.setValue(v)
                    p.setDefault(v)
        if f.input_functions is not None:
            set_function_defaults(mdata, funcs=f.input_functions)


def pipeline_preview_action(widget, callback, update=True, slc=None):
    global functions

    if len(functions) < 1:
        return None, None, None
    elif 'Reconstruction' not in [func.func_name for func in functions]:
        QtGui.QMessageBox.warning(None, 'Reconstruction method required',
                                  'You have to select a reconstruction method to run a preview')
        return None, None, None

    return construct_preview_pipeline(widget, callback, update=update, slc=slc)


def construct_preview_pipeline(widget, callback, update=True, slc=None):
    global functions, cor_offset, cor_scale

    lock_function_params(True)  # you probably do not need this anymore
    params = OrderedDict()
    funstack = []
    for func in functions:
        if not func.previewButton.isChecked() and func.func_name != 'Reconstruction':
            continue
        elif func.func_name == 'Pad' and func.paramdict()['axis'] == 2:
            n = func.paramdict()['npad']
            cor_offset = lambda x: cor_scale(x) + n
        params[func.func_name] = {func.subfunc_name: deepcopy(func.paramdict(update=update))}
        funstack.append(update_function_partial(func.partial, func.func_name, func.args_complement, widget,
                                                input_partials=func.input_partials, slc=slc))
        if func.input_functions is not None:
            params[func.func_name][func.subfunc_name]['Input Functions'] = {infunc.func_name: {infunc.subfunc_name:
                                                                            deepcopy(infunc.paramdict(update=update))}
                                                                            for infunc in func.input_functions
                                                                            if infunc is not None}
    lock_function_params(False)
    return funstack, widget.getsino(slc), partial(callback, params)


def update_function_partial(fpartial, name, argnames, datawidget, input_partials=None, slc=None, ncore=None):
    global recon_function, cor_offset, cor_scale
    kwargs = {}
    for arg in argnames:
        if arg in 'flats':
            kwargs[arg] = datawidget.getflats(slc=slc)
        if arg in 'darks':
            kwargs[arg] = datawidget.getdarks(slc=slc)
        if arg in 'ncore' and ncore is not None:
            kwargs[arg] = ncore

    if input_partials is not None:
        for pname, slices, ipartial in input_partials:
            pargs = []
            if slices is not None:
                map(pargs.append, (map(datawidget.data.fabimage.__getitem__, slices)))
            kwargs[pname] = ipartial(*pargs)
            if pname == 'center':
                if cor_offset is None:
                    cor_offset = cor_scale
                recon_function.params.child('center').setValue(kwargs[pname])
                kwargs[pname] = cor_offset(kwargs[pname])
                reset_cor()

    if kwargs:
        return partial(fpartial, **kwargs)
    else:
        return fpartial


def run_preview_recon(funstack, initializer, callback):
    if funstack is not None:
        runnable = threads.RunnableMethod(callback, reduce, (lambda f1, f2: f2(f1)), funstack, initializer)
        runnable.lock = threads.mutex
        threads.queue.put(runnable)


def run_full_recon(widget, proj, sino, out_name, out_format, nchunk, ncore, update_call=None, finish_call=None):
    global functions
    lock_function_params(True)
    partials, params = [], OrderedDict()
    for f in functions:
        params[f.subfunc_name] = deepcopy(f.paramdict(update=update))
        partials.append([f.name, deepcopy(f.partial), f.args_complement, deepcopy(f.input_partials)])
    lock_function_params(False)

    import dxchange as dx
    if out_format == 'TIFF (.tiff)':
        partials.append(('Write to file', partial(dx.write_tiff_stack, fname=out_name), [], None))
    else:
        print 'Only tiff support right now'
        return

    runnable_it = threads.RunnableIterator(update_call, _recon_iter, widget, partials, proj, sino, nchunk, ncore)
    runnable_it.emitter.sigFinished.connect(finish_call)
    threads.queue.put(runnable_it)
    return params
#TODO have current recon parameters in run console or in recon view...


def _recon_iter(datawidget, partials, proj, sino, nchunk, ncore):
    write_start = sino[0]
    total_sino = (sino[1] - sino[0] - 1) // sino[2] + 1
    nsino = (total_sino - 1) // nchunk + 1
    for i in range(nchunk):
        init = True
        start, end = i * nsino + sino[0], (i + 1) * nsino + sino[0]
        for name, fpartial, argnames, ipartials in partials:
            fpartial = update_function_partial(fpartial, name, argnames, datawidget,
                                               slc=(slice(*proj), slice(start, end, sino[2])),
                                               ncore=ncore, input_partials=ipartials)
            yield 'Running {0} on sinograms {1} to {2} from {3}...\n\n'.format(name, start, end, total_sino)
            if init:
                tomo = fpartial(datawidget.getsino(slc=(slice(*proj), slice(start, end, sino[2]))))
                shape = 2*(tomo.shape[2],)
                init = False
            elif name == 'Write to file':
                fpartial(tomo, start=write_start)
                write_start += tomo.shape[0]
            elif name == 'Reconstruction':
                # Reset input_partials to None so that centers and angle vectors are not computed in every iteration
                if ipartials is not None:
                    partials[partials.index([name, fpartial, argnames, ipartials])][3] = None

                tomo = fpartial(tomo)
                # Make sure to crop down recon if padding was used
                if tomo.shape[1:] != shape:
                    npad = tomo.shape[1] - shape[1]
                    tomo = tomo[:, npad:-npad, npad:-npad]
            else:
                tomo = fpartial(tomo)



