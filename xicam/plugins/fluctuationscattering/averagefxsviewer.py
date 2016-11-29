from PySide.QtGui import *
from PySide.QtCore import *


class averagefxsviewer(QWidget):
    sigQHover = Signal(float)
    def __init__(self,*args,**kwargs):
        super(averagefxsviewer, self).__init__()