from cx_Freeze import setup, Executable
import sys
# import scipy.sparse.csgraph._validation
sys.path.append('hipies/')


# Notes:
# Build error with scipy? edit cx_Freeze hooks.py line 548...
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
product_name = 'HipIES'

# Dependencies are automatically detected, but it might need
# fine tuning.

shortcut_table = [
    ("DesktopShortcut",  # Shortcut
     "DesktopFolder",  # Directory_
     "HipIES",  # Name
     "TARGETDIR",  # Component_
     "[TARGETDIR]hipies.exe",  # Target
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
     "HipIES",  # Name
     "TARGETDIR",  # Component_
     "[TARGETDIR]hipies.exe",  # Target
     None,  # Arguments
     None,  # Description
     None,  # Hotkey
     'C:\\Program Files (x86)\\Hipies\\icon.ico',
     # Icon [This is bad, but I can't find a better way; would have to edit msi database builder, which I'd rather not do for now]
     None,  # IconIndex
     None,  # ShowCmd
     'TARGETDIR'  # WkDir
     )
]

msi_data = {'Shortcut': shortcut_table}

buildOptions = {'packages': ['hipies', 'scipy', 'pipeline', 'daemon'],
                'includes': ['PIL', 'PySide.QtXml','scipy','h5py'],  # ,'scipy.sparse.csgraph._validation'
                'excludes': ['PyQt', 'PyQt5', 'pyqt', 'collections.sys', 'collections._weakref', 'PyQt4', 'cairo', 'tk',
                             'matplotlib', 'pyopencl', 'tcl', 'TKinter', 'tkk'], 'optimize': 2,
                'include_files': ['gui/', 'icon.ico', ('C:\\Python27\\Lib\\site-packages\\scipy\\special\\_ufuncs.pyd','_ufuncs.pyd')]}

msiOptions = {'initial_target_dir': r'[ProgramFilesFolder]\%s\%s' % (company_name, product_name)}

bdistmsiOptions = {"data": msi_data}


# ,'resources':['hipies/gui/'],'iconfile':'hipies/gui/icon.icns','includes':['PIL']

base = 'Win32GUI' if sys.platform == 'win32' else None

executables = [
    Executable('main.py', base=base, targetName='hipies.exe', icon='icon.ico', shortcutName="HipIES",
               shortcutDir="StartMenuFolder", )
]

setup(name='HipIES',
      version='1.0',
      author='Advanced Light Source',
      author_email='ronpandolfi@lbl.gov',
      description='High Performance Interactive Environment for Scattering',
      options={'build_exe': buildOptions, 'build_msi': msiOptions, 'bdist_msi': bdistmsiOptions},
      executables=executables)
