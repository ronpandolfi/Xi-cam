from PySide import QtGui, QtCore  # (the example applies equally well to PySide)
import pyqtgraph as pg
import numpy as np

app = QtGui.QApplication([])
w = QtGui.QWidget()
plot = pg.PlotWidget()

imageData = np.load('waterfall(1).npy')
TID = np.transpose(imageData)  # TID = transposes imageData is creating TID

slider = QtGui.QSlider(QtCore.Qt.Vertical, w)
slider.setGeometry(500, 500, 30, 10)
slider.setFocusPolicy(QtCore.Qt.NoFocus)

curves = []

for i in range(1, TID.shape[1]):
    curves.append(plot.plot(TID[:,i], pen=(i,TID.shape[1]*1.3)))

def getValue(value):
    print value
    for i in range(1, TID.shape[1]):
        curves[i].setData(TID[:,i]+(i*value))
        print i

w.connect(slider, QtCore.SIGNAL('valueChanged(int)'), getValue)

layout = QtGui.QGridLayout()
w.setLayout(layout)

layout.addWidget(plot, 0, 2, 2, 1)
layout.addWidget(slider, 0, 1, 2, 1)

w.show()
app.exec_()
