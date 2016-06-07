from cx_Freeze import setup, Executable
import sys
from numpy.distutils.core import Extension
import numpy as np
# import scipy.sparse.csgraph._validation
sys.path.append('xicam/')
import zmq.libzmq

# Notes:
# Build error with scipy? edit cx_Freeze hooks.py line 548...http://stackoverflow.com/questions/32432887/cx-freeze-importerror-no-module-named-scipy
# Build error with h5py? edit cx_Freeze hooks.py and comment out finder.IncludeModule('h5py.api_gen')
# Missing ufuncs? Its fine, copy numpy's lib/libifcoremd.dll and libmmd.dll into build directory...
# pyfits unsupported operand type? Comment those lines!...
# Missing h5py _errors? edit cx_Freeze hooks.py for h5py...

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
     "[TARGETDIR]xicam.exe",  # Target
     None,  # Arguments
     None,  # Description
     None,  # Hotkey
     'icon.ico',  # Icon
     None,  # IconIndex
     None,  # ShowCmd
     'TARGETDIR'  # WkDir
     ),
    ("StartMenuShortcut",  # Shortcut
     "StartMenuFolder",  # Directory_
     "Xi-cam",  # Name
     "TARGETDIR",  # Component_
     "[TARGETDIR]xicam.exe",  # Target
     None,  # Arguments
     None,  # Description
     None,  # Hotkey
     'C:\\Program Files (x86)\\Xi-cam\\icon.ico',
     # Icon [This is bad, but I can't find a better way; would have to edit msi database builder, which I'd rather not do for now]
     None,  # IconIndex
     None,  # ShowCmd
     'TARGETDIR'  # WkDir
     )
]

msi_data = {'Shortcut': shortcut_table}

buildOptions = {'packages': ['xicam', 'scipy', 'pipeline', 'daemon','zmq.backend.cython','OpenGL.platform','zmq.utils','pygments.styles'],
                'includes': ['PIL', 'PySide.QtXml','scipy','h5py','cython','zmq.backend','zmq.backend.cython','pygments.lexers.python','ipykernel.datapub'],  # ,'scipy.sparse.csgraph._validation'
                'excludes': ['PyQt', 'PyQt5', 'pyqt', 'collections.sys', 'collections._weakref', 'PyQt4', 'cairo', 'tk',
                             'matplotlib', 'pyopencl', 'tcl', 'TKinter', 'tkk'], 'optimize': 2,
                'include_files': ['gui/', 'yaml/', 'icon.ico', ('C:\\Python27\\Lib\\site-packages\\scipy\\special\\_ufuncs.pyd','_ufuncs.pyd'),zmq.libzmq.__file__]}

msiOptions = {'initial_target_dir': r'[ProgramFilesFolder]\%s\%s' % (company_name, product_name)}

bdistmsiOptions = {"data": msi_data}


# ,'resources':['xicam/gui/'],'iconfile':'xicam/gui/icon.icns','includes':['PIL']

base = 'Win32GUI' if sys.platform == 'win32' else None

executables = [
    Executable('main.py', base=base, targetName='xicam.exe', icon='icon.ico', shortcutName="Xi-cam",
               shortcutDir="StartMenuFolder", )
]

EXT = Extension(name='pipeline.cWarpImage',
                sources=['cext/cWarpImage.cc', 'cext/remesh.cc'],
                extra_compile_args=['-O3', '-ffast-math'],  # '-fopenmp',, '-I/opt/local/include'
                # extra_link_args=['-fopenmp'],
                include_dirs=[np.get_include()],

                )

setup(name='Xi-cam',
      version='1.2.3',
      author='Advanced Light Source',
      author_email='ronpandolfi@lbl.gov',
      description='High Performance Interactive Environment for Scattering',
      options={'build_exe': buildOptions, 'build_msi': msiOptions, 'bdist_msi': bdistmsiOptions},
      #ext_modules=[EXT],
      executables=executables)
