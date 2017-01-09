

block_cipher = None

import sys
import os


folder = os.getcwd()

from distutils.sysconfig import get_python_lib
from sys import platform as _platform

site_packages_path = get_python_lib()
import pyFAI
import matplotlib
import lib2to3

pyFAI_path = os.path.dirname(pyFAI.__file__)
# matplotlib_path = os.path.dirname(matplotlib.__file__)
# lib2to3_path = os.path.dirname(lib2to3.__file__)

extra_datas = [
    # ("hipgisaxs.exe", "hipgisaxs.exe"),
    ("xicam/gui", "xicam/gui"),
    ("yaml", "yaml"),
    ("icon.ico","icon.ico")
    # (os.path.join(lib2to3_path, 'Grammar.txt'), 'lib2to3/'),
    # (os.path.join(lib2to3_path, 'PatternGrammar.txt'), 'lib2to3/'),
    # ("dioptas/model/util/data/*.json", "dioptas/model/util/data"),
    # ('C:\\Python27\\Lib\\site-packages\\scipy\\special\\_ufuncs.pyd', '_ufuncs.pyd')
]

binaries = []

if _platform == "darwin":
    extra_datas.extend((
        # (os.path.join(os.path.expanduser('~'), '//anaconda/lib/libQtCore.4.dylib'), '.'),
        # (os.path.join(os.path.expanduser('~'), '//anaconda/lib/libQtWidgets.4.dylib'), '.'),
        # (os.path.join(os.path.expanduser('~'), '//anaconda/lib/libpng16.16.dylib'), '.'),
        # (os.path.join(os.path.expanduser('~'), '//anaconda/lib/libQtSvg.4.dylib'), '.'),
        # (os.path.join(os.path.expanduser('~'), '//anaconda/lib/libhdf5.10.dylib'), '.'),
        # (os.path.join(os.path.expanduser('~'), '//anaconda/lib/libhdf5_hl.10.dylib'), '.'),
    ))

sourcedirs = ['xicam','xicamlauncher','pipeline','modpkgs','client']
pyfiles = []

for path, subdirs, files in os.walk('.'):
    for name in files:
        if os.path.splitext(name)[-1]=='.py': pyfiles.append(os.path.join(path, name))


a = Analysis(['xicamlauncher/main.py','xicam/__init__.py','xicamlauncher/splash.py'],
             pathex=[folder],
             binaries=binaries,
             datas=extra_datas,
             hiddenimports=['scipy.special._ufuncs_cxx', 'skimage._shared.geometry'],
             hookspath=[],
             runtime_hooks=[],
             excludes=['FixTk', 'tcl', 'tk', '_tkinter', 'tkinter', 'Tkinter','PyQt', 'PyQt5', 'pyqt', 'collections.sys', 'collections._weakref', 'PyQt4', 'cairo', 'tk',
                             'matplotlib', 'pyopencl', 'tcl', 'TKinter', 'tkk'],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)

# remove extra packages
a.binaries = [x for x in a.binaries if not x[0].startswith("matplotlib")]
# a.binaries = [x for x in a.binaries if not x[0].startswith("zmq")]
# a.binaries = [x for x in a.binaries if not x[0].startswith("IPython")]
# a.binaries = [x for x in a.binaries if not x[0].startswith("docutils")]
# a.binaries = [x for x in a.binaries if not x[0].startswith("pytz")]
# a.binaries = [x for x in a.binaries if not x[0].startswith("wx")]
a.binaries = [x for x in a.binaries if not x[0].startswith("libQtWebKit")]
a.binaries = [x for x in a.binaries if not x[0].startswith("libQtDesigner")]
# a.binaries = [x for x in a.binaries if not x[0].startswith("PySide")]
# a.binaries = [x for x in a.binaries if not x[0].startswith("libtk")]


exclude_datas = [
    "IPython",
#   "matplotlib",
#   "mpl-data", #needs to be included
#   "_MEI",
#   "docutils",
#   "pytz",
#   "lib",
   "include",
   "sphinx",
#   ".py",
   "tests",
   "skimage",
   "alabaster",
   "boto",
   "jsonschema",
   "babel",
   "idlelib",
   "requests",
   "qt4_plugins",
   "qt5_plugins"
]

for exclude_data in exclude_datas:
    a.datas = [x for x in a.datas if exclude_data not in x[0]]


platform = ''

if _platform == "linux" or _platform == "linux2":
    platform = "Linux"
    name = "Xi-cam"
elif _platform == "win32" or _platform == "cygwin":
    platform = "Win"
    name = "Xi-cam.exe"
elif _platform == "darwin":
    platform = "Mac"
    name = "Xi-cam"

# checking whether the platform is 64 or 32 bit
if sys.maxsize > 2 ** 32:
    platform += "64"
else:
    platform += "32"

# getting the current version of Dioptas
__version__ = '0.0.1'

pyz = PYZ(a.pure, a.zipped_data,
          cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name=name,
          debug=False,
          strip=False,
          upx=True,
          console=False,
          icon="icon.ico")

coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='Xi-cam_{}_{}'.format(platform, __version__))

if _platform == "darwin":
    app = BUNDLE(coll,
                 name='Xi-cam_{}.app'.format(__version__),
icon='xicam/gui/icon.icns')