from .. import widgets
from PySide.QtCore import *

class rawviewer(widgets.dimgViewer):
    sigQHover = Signal(float)

    def replot(self):
        pass

    def mouseMoved(self, evt):
        super(rawviewer, self).mouseMoved(evt)
        if self.viewbox.sceneBoundingRect().contains(evt):
            mousePoint = self.viewbox.mapSceneToView(evt)
            x = mousePoint.x()
            y = mousePoint.y()
            self.sigQHover.emit(self.getq(x,y))
