import re
import platform

from PySide import QtCore

## Test whether we have Qt >= 4.7 
have_qt47 = True
versionReq = [4, 7]
QtVersion = QtCore.__version__ 
m = re.match(r'(\d+)\.(\d+).*', QtVersion)
if m is not None and list(map(int, m.groups())) < versionReq:
    have_qt47 = False

## Test whether we are using Windows
if platform.system() == 'Windows':
    have_windows = True
else:
    have_windows = False

