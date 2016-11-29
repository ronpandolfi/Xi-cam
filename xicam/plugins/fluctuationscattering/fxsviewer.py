from PySide.QtGui import *
from PySide.QtCore import *

class fxsviewer(QWidget):
    sigQHover = Signal(float)
    def __init__(self,*args,**kwarsg):
        super(fxsviewer, self).__init__()

