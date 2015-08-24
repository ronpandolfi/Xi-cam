import os
import string
from PySide import QtGui
import sys

def similarframe(path, N):
    """
    Get the file path N ahead (or behind) the provided path frame.
    """
    try:
        framenum = os.path.splitext(os.path.basename(path).split('_')[-1])[0]
        prevframenum = int(framenum) + N
        prevframenum = '{:0>5}'.format(prevframenum)
        return string.replace(path, framenum, prevframenum)
    except ValueError:
        print('No earlier frame found for ' + path)
        return None


def path2nexus(path):
    """
    Get the path to corresponding nexus file
    """
    return os.path.splitext(path)[0] + '.nxs'


def getRoot():
    if sys.platform == 'linux2':
        return '/'
    elif sys.platform == 'darwin':
        return '/Volumes'
    elif sys.platform == 'win32':
        return QtGui.QFileSystemModel().myComputer()
    else:
        print 'WARNING: Unknown platform "' + sys.platform + '"'

    return None
