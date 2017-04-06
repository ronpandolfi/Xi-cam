from cx_Freeze import setup, Executable
import opcode
import sys
from numpy.distutils.core import Extension
import numpy as np
# import scipy.sparse.csgraph._validation
sys.path.append('xicam/')
try:
    import zmq.libzmq # not available on linux (?)
except:
    zmq = None

import pyFAI
import os

# Notes:
# Build error with scipy? edit cx_Freeze hooks.py line 548...http://stackoverflow.com/questions/32432887/cx-freeze-importerror-no-module-named-scipy
# Build error with h5py? edit cx_Freeze hooks.py and comment out finder.IncludeModule('h5py.api_gen')
# Missing ufuncs? Its fine, copy numpy's lib/libifcoremd.dll and libmmd.dll into build directory...
# pyfits unsupported operand type? Comment those lines!...
# Missing h5py _errors? edit cx_Freeze hooks.py for h5py...
# Pyqtgraph plots displaying wrong? PySide 1.2.4 seems broken on windows; install 1.2.2 from .exe
# lmfit is not a directory? uninstall/reinstall it (replaces .egg)
# numpy .format missing? add this to hooks.py:
    # def load_numpy_lib(finder, module):
    #     finder.IncludeModule('numpy.lib.format')
# distutils __version__ missing? replace h5py.version version number over engineering with a tuple
    # version = (2,6,0)

# H5PY FIX:
# def load_h5py(finder, module):
#    """h5py module has a number of implicit imports"""
#    finder.IncludeModule('h5py.defs')
#    finder.IncludeModule('h5py.utils')
#    finder.IncludeModule('h5py._proxy')
#    try:
#       finder.IncludeModule('h5py._errors')
#       finder.IncludeModule('h5py.h5ac')
#    except:
#       pass
#    try:
#       finder.IncludeModule('h5py.api_gen')
#    except:
#       pass

company_name = 'Advanced Light Source'
product_name = 'Xi-cam'

# Dependencies are automatically detected, but it might need
# fine tuning.

shortcut_table = [
    ("DesktopShortcut",  # Shortcut
     "DesktopFolder",  # Directory_
     "Xi-cam",  # Name
     "TARGETDIR",  # Component_
     "[TARGETDIR]Xi-cam.exe",  # Target
     None,  # Arguments
     None,  # Description
     None,  # Hotkey
     'C:\\Program Files\\Xi-cam\\icon.ico',  # Icon
     None,  # IconIndex
     None,  # ShowCmd
     'TARGETDIR'  # WkDir
     ),
    ("StartMenuShortcut",  # Shortcut
     "StartMenuFolder",  # Directory_
     "Xi-cam",  # Name
     "TARGETDIR",  # Component_
     "[TARGETDIR]Xi-cam.exe",  # Target
     None,  # Arguments
     None,  # Description
     None,  # Hotkey
     'C:\\Program Files\\Xi-cam\\icon.ico',
     # Icon [This is bad, but I can't find a better way; would have to edit msi database builder, which I'd rather not do for now]
     None,  # IconIndex
     None,  # ShowCmd
     'TARGETDIR'  # WkDir
     )
]

msi_data = {'Shortcut': shortcut_table}

def include_OpenGL():
    path_base = "C:\\Python27-64\\Lib\\site-packages\\OpenGL"
    skip_count = len(path_base)
    zip_includes = [(path_base, "OpenGL")]
    for root, sub_folders, files in os.walk(path_base):
        for file_in_root in files:
            zip_includes.append(
                ("{}".format(os.path.join(root, file_in_root)),
                 "{}".format(os.path.join("OpenGL", root[skip_count+1:], file_in_root))
                 )
            )
    return zip_includes


getglobalpkg = lambda name: (os.path.join(os.path.dirname(opcode.__file__), name),name)

buildOptions = {'packages': ['xicam', 'xicamlauncher', 'scipy', 'pipeline', 'daemon','zmq.backend.cython','OpenGL',
                             'OpenGL.platform','zmq.utils','pygments.styles','idna',
                             'pyqtgraph','distutils','IPython','numpy','cryptography','paramiko','tornado','distributed','cffi'],
                'includes': ['PIL', 'PySide.QtXml','PySide.QtNetwork','PySide.QtWebKit','scipy','h5py','cython','zmq.backend','zmq.backend.cython',
                             'pygments.lexers.python','ipykernel.datapub',
                             'cryptography.hazmat.backends.openssl','cryptography.hazmat.backends.commoncrypto',
                             'numpy.core._methods','numpy.lib.format'
                             ],  # ,'scipy.sparse.csgraph._validation'
                'excludes': ['site','PyQt', 'PyQt5', 'pyqt', 'collections.sys', 'collections._weakref', 'PyQt4', 'cairo', 'tk',
                             'matplotlib', 'pyopencl', 'tcl', 'TKinter', 'tkk'], 'optimize': 0,
                'include_files': [getglobalpkg('distutils'),getglobalpkg('site.py'),'tiff.dll','hipgisaxs.exe',
                                  ('xicam/gui/','xicam/gui/'), 'yaml/', 'icon.ico',
                                  ('C:\\Python27-64\\Lib\\site-packages\\scipy\\special\\_ufuncs.pyd','_ufuncs.pyd'),
                                  zmq.libzmq.__file__,pyFAI.__path__[0]],
                'zip_includes': include_OpenGL(),
                'include_msvcr': True
                }

if zmq is not None: buildOptions['include_files'].append(zmq.libzmq.__file__)

msiOptions = {'initial_target_dir': r'[ProgramFilesFolder]\%s\%s' % (company_name, product_name)}

bdistmsiOptions = {"data": msi_data}


# ,'resources':['xicam/gui/'],'iconfile':'xicam/gui/icon.icns','includes':['PIL']

base = 'Win32GUI' if sys.platform == 'win32' else None

executables = [
    Executable('run_xicam.py', base=base, targetName='Xi-cam.exe', icon='icon.ico', shortcutName="Xi-cam", # DO NOT REMOVE '.exe' FOR WINDOWS BUILDS
               shortcutDir="StartMenuFolder")
]

EXT = Extension(name='pipeline.cWarpImage',
                sources=['cext/cWarpImage.cc', 'cext/remesh.cc'],
                extra_compile_args=['-O3', '-ffast-math'],  # '-fopenmp',, '-I/opt/local/include'
                # extra_link_args=['-fopenmp'],
                include_dirs=[np.get_include()],

                )

setup(name='Xi-cam',
      version='1.2.19',
      author='Advanced Light Source',
      author_email='ronpandolfi@lbl.gov',
      description='High Performance Interactive Environment for Scattering',
      options={'build_exe': buildOptions, 'build_msi': msiOptions, 'bdist_msi': bdistmsiOptions},
      #ext_modules=[EXT],
      executables=executables)
