import numpy as np
from PySide.QtGui import *
from PySide.QtCore import *
import pyqtgraph as pg
from pyqtgraph import parametertree as pt
from astropy.modeling import models, Fittable1DModel, Parameter, fitting


class FitParameter(pt.parameterTypes.GroupParameter):
    sigRangeChanged = Signal()
    def __init__(self,plotwidget):
        super(FitParameter, self).__init__(name='Fitting', type='group')
        self._plotwidget=plotwidget

        self.rangemin = 0
        self.rangemax = 0
        self.showrange = False

        self.rangeROI = pg.LinearRegionItem([0, 1], pg.LinearRegionItem.Vertical, brush=pg.mkBrush([128, 0, 128, 32]))
        for line in self.rangeROI.lines: line.setPen(pg.mkPen(color=[128, 0, 128], width=2))
        self.rangeROI.setVisible(False)
        self.plotwidget.addItem(self.rangeROI)

        self.changeModel(None, models.Lorentz1D)

    @property
    def plotwidget(self):
        return self._plotwidget() if callable(self._plotwidget) else self._plotwidget

    @Slot(object,object)
    def changeModel(self,sender,model):
        self.modelParam = ModelParameter(model, self)
        modeltypeparam = pt.Parameter.create(name='Model', type='list', values=usefulmodels,value=model)
        rangeminparam = pt.Parameter.create(name='Range min.', type='float', value=self.rangemin)
        rangemaxparam = pt.Parameter.create(name='Range max.', type='float', value=self.rangemax)
        showrangeparam = pt.Parameter.create(name='Show range', type='bool', value=self.showrange)

        modeltypeparam.sigValueChanged.connect(self.changeModel)
        self.sigRangeChanged.connect(self.updateROI)
        self.rangeROI.sigRegionChangeFinished.connect(self.updateRange)

        self.clearChildren()
        self.addChildren([modeltypeparam,
                          rangeminparam,
                          rangemaxparam,
                          showrangeparam,
                          self.modelParam])

        self.param('Range min.').sigValueChanged.connect(self.sigRangeChanged)
        self.param('Range max.').sigValueChanged.connect(self.sigRangeChanged)
        self.param('Show range').sigValueChanged.connect(self.sigRangeChanged)


    def updateROI(self):
        self.rangemin = self.param('Range min.').value()
        self.rangemax = self.param('Range max.').value()
        self.showrange = self.param('Show range').value()

        self.rangeROI.sigRegionChangeFinished.disconnect(self.updateRange)
        self.rangeROI.setRegion((self.rangemin, self.rangemax))
        self.rangeROI.setVisible(self.showrange)
        self.rangeROI.sigRegionChangeFinished.connect(self.updateRange)
        self.plotwidget.addItem(self.rangeROI)


    def updateRange(self):
        self.rangemin, self.rangemax = self.rangeROI.getRegion()

        self.param('Range min.').setValue(self.rangemin,blockSignal=self.sigRangeChanged)
        self.param('Range max.').setValue(self.rangemax,blockSignal=self.sigRangeChanged)





class ModelParameter(pt.Parameter):

    def __init__(self,model,parentTree):
        self.parentTree=parentTree
        self.model=model
        inputs={'name':'Inputs','type':'group','children':[]}


        outputs={'name':'Outputs','type':'group','children':[],'readonly':True}
        for param in model.param_names:
            inputs['children'].append({'name':param,'type':'float'})
            outputs['children'].append({'name': param, 'type': 'float'})

        fitAction = {'name':'Run Fit','type':'action'}

        children = [inputs, outputs,fitAction]
        super(ModelParameter, self).__init__(name=model.name,
                                             type='group',
                                             children=children)

        self.param('Run Fit').sigActivated.connect(self.RunFit)



    def RunFit(self):
        for item in self.parentTree.plotwidget.listDataItems():
            if hasattr(item,'isFit'):
                self.parentTree.plotwidget.removeItem(item)
                self.parentTree.plotwidget.removeItem(item.display_text)
                continue
            x,y = item.getData()
            limits = self.parentTree.param('Range min.').value(),self.parentTree.param('Range max.').value()
            y=y[np.logical_and(limits[0]<x,x<limits[1])]
            x=x[np.logical_and(limits[0]<x,x<limits[1])]
            inputs = dict((param.opts['name'],param.value()) for param in self.param('Inputs').children())
            g_init = self.model(**inputs)
            fit_g = fitting.LevMarLSQFitter()
            g = fit_g(g_init, x, y)
            txt=''
            for paramname in g.param_names:
                self.param('Outputs').param(paramname).setValue(getattr(g,paramname).value)
                txt+=paramname +': '+str(getattr(g,paramname).value)+'\n'

            txt=txt[:-1] # remove trailing new line

            # Plot fit in similar style
            pen = pg.mkPen(item.opts['pen'])
            pen.setStyle(Qt.DashLine)
            curve = self.parentTree.plotwidget.plot(x,g(x),pen=pen)
            curve.isFit=True
            curve.display_text = pg.TextItem(text='Test', color=(255, 255, 255), anchor=(0, 1),
                                            fill=pg.mkBrush(255, 127, 0, 100))
            curve.display_text.hide()
            self.parentTree.plotwidget.addItem(curve.display_text)
            def hover(event):
                curve.display_text.setPos(event.pos())
                curve.display_text.setVisible(not event.exit)
                curve.display_text.setText(txt)
            curve.curve.hoverEvent=hover



class LogNormal(models.Gaussian1D):

    @staticmethod
    def evaluate(x, amplitude, mean, stddev):
        return models.Gaussian1D.evaluate(np.log(x),amplitude,np.log(mean),stddev/mean)

    fit_deriv = None


usefulmodels = [models.Gaussian1D,models.Voigt1D,models.Lorentz1D,LogNormal]
usefulmodels = dict((model.name,model) for model in usefulmodels)

## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    # ######## SAVE THIS FOR DEBUGGING SEG FAULTS; issues are usually doing something outside the gui thread
    # import sys
    #
    #
    # def trace(frame, event, arg):
    #     print "%s, %s:%d" % (event, frame.f_code.co_filename, frame.f_lineno)
    #     return trace
    #
    #
    # sys.settrace(trace)

    app = QApplication([])

    ## Create window with ImageView widget
    win = QMainWindow()
    win.resize(800, 800)

    p = pg.PlotWidget()
    range = np.arange(0,50,1)
    p.plot(range,models.Gaussian1D.evaluate(range,10,30,5),pen={'color':'r','width':2})
    p.show()

    fitter = FitParameter(p)
    tree = pt.ParameterTree()
    tree.setParameters(fitter,showTop=False)
    win.setCentralWidget(tree)
    win.setWindowTitle('Fitting')
    win.show()

    QApplication.instance().exec_()
