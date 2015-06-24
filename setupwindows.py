from cx_Freeze import setup, Executable
import sys
import scipy.sparse.csgraph._validation
sys.path.append('hipies/')

company_name = 'Advanced Light Source'
product_name = 'HipIES'

# Dependencies are automatically detected, but it might need
# fine tuning.

shortcut_table = [
    ("DesktopShortcut",        # Shortcut
     "DesktopFolder",          # Directory_
     "HipIES",           # Name
     "TARGETDIR",              # Component_
     "[TARGETDIR]hipies.exe",# Target
     None,                     # Arguments
     None,                     # Description
     None,                     # Hotkey
     'icon.ico',                     # Icon
     None,                     # IconIndex
     None,                     # ShowCmd
     'TARGETDIR'               # WkDir
     ),
    ("StartMenuShortcut",        # Shortcut
     "StartMenuFolder",          # Directory_
     "HipIES",           # Name
     "TARGETDIR",              # Component_
     "[TARGETDIR]hipies.exe",# Target
     None,                     # Arguments
     None,                     # Description
     None,                     # Hotkey
     'C:\\Program Files (x86)\\Hipies\\icon.ico', # Icon [This is bad, but I can't find a better way; would have to edit msi database builder, which I'd rather not do for now]
     None,                     # IconIndex
     None,                     # ShowCmd
     'TARGETDIR'               # WkDir
     )
    ]

msi_data = {'Shortcut':shortcut_table}

buildOptions = {'packages': ['hipies','scipy','pipeline','daemon'],
                'includes': ['PIL', 'PySide.QtXml', 'PySide','scipy.sparse.csgraph._validation'],
                'excludes': ['PyQt', 'PyQt5', 'pyqt', 'collections.sys', 'collections._weakref', 'PyQt4', 'cairo', 'tk',
                             'matplotlib','pyopencl','tcl','TKinter','tkk'], 'optimize': 2,
                'include_files': ['gui/','icon.ico']}

msiOptions = {'initial_target_dir': r'[ProgramFilesFolder]\%s\%s' % (company_name, product_name)}

bdistmsiOptions = {"data": msi_data}


# ,'resources':['hipies/gui/'],'iconfile':'hipies/gui/icon.icns','includes':['PIL']

base = 'Win32GUI' if sys.platform == 'win32' else None

executables = [
    Executable('main.py', base=base, targetName='hipies.exe',icon='gui/icon.ico',shortcutName="HipIES",shortcutDir="StartMenuFolder",)
]

setup(name='HipIES',
      version='0.6',
      author='Advanced Light Source',
      author_email='ronpandolfi@lbl.gov',
      description='High Performance Interactive Environment for Scattering',
      options={'build_exe': buildOptions,'build_msi': msiOptions,'bdist_msi':bdistmsiOptions},
      executables=executables)
