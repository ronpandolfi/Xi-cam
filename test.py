# --coding: utf-8 --

import numpy as np
from PySide import QtGui, QtCore
import pyqtgraph as pg

if __name__ == '__main__':
    from paws import api
    paw = api.start()
    paw.load_from_wfl("/home/rp/Downloads/abc.wfl")
    paw.execute()
    print(paw)