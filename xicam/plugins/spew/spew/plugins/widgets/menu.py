# -*- coding: utf-8 -*-
"""
@author: rpandolfi
"""
from PySide import QtGui


class pluginModeWidget(QtGui.QWidget):
    def __init__(self, plugins):
        super(pluginModeWidget, self).__init__()
        l = QtGui.QHBoxLayout()
        l.setContentsMargins(0, 0, 0, 0)
        l.setSpacing(0)
        self.setLayout(l)
        self.font = QtGui.QFont()
        self.font.setPointSize(16)
        self.plugins = plugins
        self.setStyleSheet('background-color:#111111;')
        self.reload()

    def reload(self):
        w = self.layout().takeAt(0)
        while w:
            w.widget().deleteLater()
            del w
            w = self.layout().takeAt(0)

        for key, plugin in self.plugins.items():
            if plugin.enabled:
                if plugin.instance.hidden:
                    continue

                button = QtGui.QPushButton(plugin.name)
                button.setFlat(True)
                button.setFont(self.font)
                button.setProperty('isMode', True)
                button.setAutoFillBackground(False)
                button.setCheckable(True)
                button.setAutoExclusive(True)
                button.clicked.connect(plugin.activate)
                if plugin is self.plugins.values()[0]:
                    button.setChecked(True)
                self.layout().addWidget(button)
                label = QtGui.QLabel('|')
                label.setMaximumWidth(5)
                label.setFont(self.font)
                label.setStyleSheet('background-color:#111111;')
                self.layout().addWidget(label)

        self.layout().takeAt(self.layout().count() - 1).widget().deleteLater()  # Delete the last pipe symbol