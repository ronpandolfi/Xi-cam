# --coding: utf-8 --

import numpy as np
from PySide import QtGui, QtCore
import pyqtgraph as pg

if __name__ == '__main__':
    import sys

    app = QtGui.QApplication(sys.argv)

    imageData = np.load('/home/rp/data/YL1031/waterfall.npy')
    TID = np.transpose(imageData)  # TID = transposes imageData is creating TID

    plt = pg.PlotItem()
    plt.setLabel('left', "Time", units='s')
    plt.setLabel('bottom', 'q (Å⁻¹)')
    plt.axes['left']['item'].setZValue(10)
    plt.setAspectLocked(True)

    view = pg.ImageView(view=plt)
    view.setImage(TID,pos=[0,0])
    view.show()

    sys.exit(app.exec_())



