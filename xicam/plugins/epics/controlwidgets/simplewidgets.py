from PySide.QtGui import *
from PySide.QtCore import *

class placeHolderSpinBox(QDoubleSpinBox):


    def __init__(self,button):
        super(placeHolderSpinBox, self).__init__()
        self.lastacceptedvalue = self.value()
        button.clicked.connect(self.setnewlast)
        self.button = button

    def setnewlast(self):
        self.lastacceptedvalue = self.value()

    def focusOutEvent(self,event):
        event.accept()
        self.setValue(self.lastacceptedvalue)
        self.setStyleSheet('color:gray')
        self.update()

    def focusInEvent(self,event):
        event.accept()
        self.setStyleSheet('')
        self.update()

    def keyPressEvent(self, e):
        super(placeHolderSpinBox, self).keyPressEvent(e)
        if e.key() == Qt.Key_Return or e.key() == Qt.Key_Enter:
            self.button.clicked.emit()
            e.accept()

