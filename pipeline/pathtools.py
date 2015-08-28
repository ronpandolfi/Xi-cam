import os
import string
from PySide import QtGui
import sys
import re

def similarframe(path, N):
    """
    Get the file path N ahead (or behind) the provided path frame.
    """
    try:
        expr = '(?<=_)[\d]+(?=[_.])'
        frame = re.search(expr, os.path.basename(path)).group(0)
        leadingzeroslen=len(frame)
        framenum = int(frame)
        prevframenum = int(framenum) + N
        prevframenum = '{:0>{}}'.format(prevframenum,leadingzeroslen)
        return re.sub(expr, prevframenum, path)
    except ValueError:
        print 'No earlier frame found for ' + path + ' with ' + N
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
