from pyqtgraph.graphicsItems import GradientEditorItem
from collections import OrderedDict

GradientEditorItem.__dict__['Gradients'] = OrderedDict([
    ('thermal', {'ticks': [(0.3333, (185, 0, 0, 255)), (0.6666, (255, 220, 0, 255)), (1, (255, 255, 255, 255)), (0, (0, 0, 0, 255))], 'mode': 'rgb'}),
    ('flame', {'ticks': [(0.2, (7, 0, 220, 255)), (0.5, (236, 0, 134, 255)), (0.8, (246, 246, 0, 255)), (1.0, (255, 255, 255, 255)), (0.0, (0, 0, 0, 255))], 'mode': 'rgb'}),
    ('yellowy', {'ticks': [(0.0, (0, 0, 0, 255)), (0.2328863796753704, (32, 0, 129, 255)), (0.8362738179251941, (255, 255, 0, 255)), (0.5257586450247, (115, 15, 255, 255)), (1.0, (255, 255, 255, 255))], 'mode': 'rgb'} ),
    ('bipolar', {'ticks': [(0.0, (0, 255, 255, 255)), (1.0, (255, 255, 0, 255)), (0.5, (0, 0, 0, 255)), (0.25, (0, 0, 255, 255)), (0.75, (255, 0, 0, 255))], 'mode': 'rgb'}),
    ('viridis', {'ticks': [(0.0, (68, 1, 84, 255)), (0.25, (58, 82, 139, 255)), (0.5, (32, 144, 140, 255)), (0.75, (94, 201, 97, 255)), (1.0, (253, 231, 36, 255))], 'mode': 'rgb'}),
    ('inferno', {'ticks': [(0.0, (0, 0, 3, 255)), (0.25, (87, 15, 109, 255)), (0.5, (187, 55, 84, 255)), (0.75, (249, 142, 8, 255)), (1.0, (252, 254, 164, 255))], 'mode': 'rgb'}),
    ('plasma', {'ticks': [(0.0, (12, 7, 134, 255)), (0.25, (126, 3, 167, 255)), (0.5, (203, 71, 119, 255)), (0.75, (248, 149, 64, 255)), (1.0, (239, 248, 33, 255))], 'mode': 'rgb'}),
    ('magma', {'ticks': [(0.0, (0, 0, 3, 255)), (0.25, (80, 18, 123, 255)), (0.5, (182, 54, 121, 255)), (0.75, (251, 136, 97, 255)), (1.0, (251, 252, 191, 255))], 'mode': 'rgb'}),
    ('spectrum', {'ticks': [(1.0, (255, 0, 255, 255)), (0.0, (255, 0, 0, 255))], 'mode': 'hsv'}),
    ('cyclic', {'ticks': [(0.0, (255, 0, 4, 255)), (1.0, (255, 0, 0, 255))], 'mode': 'hsv'}),
    ('greyclip', {'ticks': [(0.0, (0, 0, 0, 255)), (0.99, (255, 255, 255, 255)), (1.0, (255, 0, 0, 255))], 'mode': 'rgb'}),
    ('grey', {'ticks': [(0.0, (0, 0, 0, 255)), (1.0, (255, 255, 255, 255))], 'mode': 'rgb'}),
])

from pyqtgraph.exporters import ImageExporter
from pyqtgraph.Qt import QtGui, QtCore, QtSvg, USE_PYSIDE
from pyqtgraph import functions as fn
import numpy as np

class fixedexporter(ImageExporter):
    def export(self, fileName=None, toBytes=False, copy=False):
        if fileName is None and not toBytes and not copy:
            if USE_PYSIDE:
                filter = ["*." + str(f) for f in QtGui.QImageWriter.supportedImageFormats()]
            else:
                filter = ["*." + bytes(f).decode('utf-8') for f in QtGui.QImageWriter.supportedImageFormats()]
            preferred = ['*.png', '*.tif', '*.jpg']
            for p in preferred[::-1]:
                if p in filter:
                    filter.remove(p)
                    filter.insert(0, p)
            self.fileSaveDialog(filter=filter)
            return

        targetRect = QtCore.QRect(0, 0, self.params['width'], self.params['height'])
        sourceRect = self.getSourceRect()

        # self.png = QtGui.QImage(targetRect.size(), QtGui.QImage.Format_ARGB32)
        # self.png.fill(pyqtgraph.mkColor(self.params['background']))
        w, h = self.params['width'], self.params['height']
        if w == 0 or h == 0:
            raise Exception("Cannot export image with size=0 (requested export size is %dx%d)" % (w, h))
        bg = np.empty((int(self.params['width']), int(self.params['height']), 4), dtype=np.ubyte)
        color = self.params['background']
        bg[:, :, 0] = color.blue()
        bg[:, :, 1] = color.green()
        bg[:, :, 2] = color.red()
        bg[:, :, 3] = color.alpha()
        self.png = fn.makeQImage(bg, alpha=True)

        ## set resolution of image:
        origTargetRect = self.getTargetRect()
        resolutionScale = targetRect.width() / origTargetRect.width()
        # self.png.setDotsPerMeterX(self.png.dotsPerMeterX() * resolutionScale)
        # self.png.setDotsPerMeterY(self.png.dotsPerMeterY() * resolutionScale)

        painter = QtGui.QPainter(self.png)
        # dtr = painter.deviceTransform()
        try:
            self.setExportMode(True, {'antialias': self.params['antialias'], 'background': self.params['background'],
                                      'painter': painter, 'resolutionScale': resolutionScale})
            painter.setRenderHint(QtGui.QPainter.Antialiasing, self.params['antialias'])
            self.getScene().render(painter, QtCore.QRectF(targetRect), QtCore.QRectF(sourceRect))
        finally:
            self.setExportMode(False)
        painter.end()

        if copy:
            QtGui.QApplication.clipboard().setImage(self.png)
        elif toBytes:
            return self.png
        else:
            self.png.save(fileName)
fixedexporter.register()