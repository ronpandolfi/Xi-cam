# --coding: utf-8 --

from PySide.QtGui import *
from PySide.QtCore import *
from rawviewer import rawviewer
from fxsviewer import fxsviewer
from averagefxsviewer import averagefxsviewer
import numpy as np

# TODO: viewers will emit qhover signal (q), which must be connected to fxsmainwidget qhover, which emits qhover with 1-D curve

class fxsmainwidget(QTabWidget):
    sigQHover = Signal(object)
    def __init__(self,*args,**kwargs):
        super(fxsmainwidget, self).__init__()

        self.setTabPosition(self.TabPosition.South)
        self.setTabShape(self.Triangular)

        self.rawviewer = rawviewer(*args,**kwargs)
        self.fxsviewer = fxsviewer(*args,**kwargs)
        self.averagefxsviewer = averagefxsviewer(*args,**kwargs)

        self.rawviewer.sigQHover.connect(self.qHover)
        self.fxsviewer.sigQHover.connect(self.qHover)
        self.averagefxsviewer.sigQHover.connect(self.qHover)

        self.addTab(self.rawviewer,'RAW')
        self.addTab(self.averagefxsviewer,u'〈f(q,q,χ)〉')
        self.addTab(self.fxsviewer,u'〈f(q,q,χ)〉')

        self.plotwidget=kwargs['plotwidget']
        self.sigQHover.connect(self.plot)

        self.curve = self.plotwidget.plot()

        self.updateCalculation()

    def updateCalculation(self):
        # self.showFXS()
        pass

    def showFXS(self):
        # self.fxsviewer.setImage(data)
        # self.averagefxsviewer.setImage(data)
        pass

    def qHover(self,q):
        self.sigQHover.emit([q,q,q,q])
        pass

    def plot(self,*args,**kwargs):
        self.curve.setData(*args,**kwargs)