import os
from PySide import QtCore
from xicam.widgets import featurewidgets as fw
from pyqtgraph.parametertree import Parameter
from filters import POCLFilter
from xicam.threads import RunnableMethod
import ClAttributes as clattr
import pyopencl as cl
import importer

class FilterManager(fw.FeatureManager):

    sigFilterAdded = QtCore.Signal()

    def __init__(self, list_layout, form_layout, function_widgets=None, blank_form=None):
        super(FilterManager, self).__init__(list_layout, form_layout, feature_widgets=function_widgets,
                                              blank_form=blank_form)

    def updateFilterMasks(self, mask_dict):
        """
        For each filterwidget, update its masks to include everything in image_masks

        Parameters
        ----------
        mask_dict: dict
            dictionary of path-f3dviewer pairs representing all potential masks usable in filter pipeline
        """

        for feature in self.features:
                for child in feature.params.children():
                    try:
                        if child.name() == 'Mask':
                            for param in feature.details['Parameters']:
                                if "Mask" in param.itervalues():
                                    default = feature.params.child("Mask").value()
                                    masks = param['values']
                                    for path in mask_dict.iterkeys():
                                        if os.path.basename(path) not in masks: masks.append(os.path.basename(path))
                                    feature.form.clear()
                                    feature.params = Parameter.create(name=feature.name,
                                                children=feature.details['Parameters'], type='group')
                                    feature.form.setParameters(feature.params, showTop=True)
                                    feature.reconnectDefaults()

                                    # set parameter back to previous value
                                    feature.params.child('Mask').setValue(default)

                    except AttributeError: pass

    def addFilter(self, name):

        filter_widget = POCLFilter.POCLFilter(name)
        self.addFeature(filter_widget)
        self.sigFilterAdded.emit()
        return filter_widget

    #other helper functions involving getting filter information will be added here


    def getPipeline(self):

        # return some representation of the pipeline, other necessary things

        pipeline = []
        # for feature in self.features:
        #     # pipeline.append(feature.filter)
        #     pipeline.append(feature)
        #
        # return pipeline
        return self.features

    def getPipelineDict(self):

        # return dictionary representation of filters in pipeline + their parameters

        pipeline = {}
        for feature in self.features:
            stack_dict = feature.stack_dict
            stack_dict.pop('Name')
            pipeline[feature.name] = stack_dict
        return pipeline

    def run(self):
        pass

class RunnablePOCLFilter(RunnableMethod):

    def __init__(self, method, method_args=(), method_kwargs={}, callback_slot=None,
                 finished_slot=None, except_slot=None, default_exhandle=True, priority=0, lock=None):
        super(RunnablePOCLFilter, self).__init__(method, method_args=method_args, method_kwargs=method_kwargs,
                  callback_slot=callback_slot, finished_slot=finished_slot, except_slot=except_slot,
                  default_exhandle=default_exhandle, priority=priority, lock=lock)










