# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2015, Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
# -----------------------------------------------------------------------------
# vispy: gallery 2
#
# Adapted for use as a widget by Ron Pandolfi
# volumeViewer.getHistogram method borrowed from PyQtGraph

"""
Example volume rendering

Controls:

* 1  - toggle camera between first person (fly), regular 3D (turntable) and
       arcball
* 2  - toggle between volume rendering methods
* 3  - toggle between stent-CT / brain-MRI image
* 4  - toggle between colormaps
* 0  - reset cameras
* [] - decrease/increase isosurface threshold

With fly camera:

* WASD or arrow keys - move around
* SPACE - brake
* FC - move up-down
* IJKL or mouse - look around
"""

from PySide import QtGui
from vispy import scene

from pipeline import loader, msg
# TODO refactor general widgets to be part of plugins.widgets to be shared in a more organized fashion
from xicam.widgets.volumeviewers import VolumeVisual, VolumeViewer
from xicam.widgets.imageviewers import StackViewer


class ThreeDViewer(QtGui.QWidget, ):
    def __init__(self, paths, parent=None):
        super(ThreeDViewer, self).__init__(parent=parent)

        self.combo_box = QtGui.QComboBox(self)
        self.combo_box.addItems(['Image Stack', '3D Volume'])
        self.stack_viewer = StackViewer()
        self.volume_viewer = VolumeViewer()

        self.view_stack = QtGui.QStackedWidget(self)
        self.view_stack.addWidget(self.stack_viewer)
        self.view_stack.addWidget(self.volume_viewer)

        hlayout = QtGui.QHBoxLayout()
        self.subsample_spinbox = QtGui.QSpinBox()
        self.subsample_label = QtGui.QLabel('Subsample Level:')
        self.loadVolumeButton = QtGui.QToolButton()
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("xicam/gui/icons_45.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.loadVolumeButton.setIcon(icon)
        self.loadVolumeButton.setToolTip('Generate Volume')
        hlayout.addWidget(self.combo_box)
        hlayout.addSpacing(2)
        hlayout.addWidget(self.subsample_label)
        hlayout.addWidget(self.subsample_spinbox)
        hlayout.addWidget(self.loadVolumeButton)
        hlayout.addStretch()
        layout = QtGui.QVBoxLayout(self)
        layout.addLayout(hlayout)
        layout.addWidget(self.view_stack)

        self.subsample_spinbox.hide()
        self.subsample_spinbox.setValue(8)
        self.subsample_label.hide()
        self.loadVolumeButton.hide()

        self.stack_image = loader.StackImage(paths)
        self.volume = None
        self.stack_viewer.setData(self.stack_image)
        self.combo_box.activated.connect(self.view_stack.setCurrentIndex)
        self.view_stack.currentChanged.connect(self.toggleInputs)
        self.loadVolumeButton.clicked.connect(self.loadVolume)

    def toggleInputs(self, index):
        if self.view_stack.currentWidget() is self.volume_viewer:
            self.subsample_label.show()
            self.subsample_spinbox.show()
            self.loadVolumeButton.show()
        else:
            self.subsample_label.hide()
            self.subsample_spinbox.hide()
            self.loadVolumeButton.hide()

    def loadVolume(self):
        msg.showMessage('Generating volume...', timeout=5)
        level = self.subsample_spinbox.value()
        self.volume = self.stack_image.asVolume(level=level)
        self.volume_viewer.setVolume(vol=self.volume, slicevol=False)
        msg.clearMessage()
