from xicam.widgets import featurewidgets as fw
from filters import JOCLFilter
import importer

class FilterManager(fw.FeatureManager):

    def __init__(self, list_layout, form_layout, function_widgets=None, blank_form=None):
        super(FilterManager, self).__init__(list_layout, form_layout, feature_widgets=function_widgets,
                                              blank_form=blank_form)

    def updateFilterMasks(self, image_masks):
        pass

        """
        For each filterwidget, update its masks to include everything in image_masks
        """

    def addFilter(self, name):

        filter_widget = JOCLFilter.JOCLFilter(name)
        # func_widget.sigTestRange.connect(self.testParameterRange)
        self.addFeature(filter_widget)
        return filter_widget



    # def loadFilters(self, filters=importer.filters):
    #
    #     self.removeAllFeatures()
    #     for filter, options in filters.iteritems():
    #         for subfunc in subfuncs:
    #             funcWidget = self.addFunction(func, subfunc, package=reconpkg.packages[config_dict[subfunc][1]])
    #             if 'Enabled' in subfuncs[subfunc] and not subfuncs[subfunc]['Enabled']:
    #                 funcWidget.enabled = False
    #             if 'Parameters' in subfuncs[subfunc]:
    #                 for param, value in subfuncs[subfunc]['Parameters'].iteritems():
    #                     child = funcWidget.params.child(param)
    #                     child.setValue(value)
    #                     if setdefaults:
    #                         child.setDefault(value)
    #             if 'Input Functions' in subfuncs[subfunc]:
    #                 for param, ipfs in subfuncs[subfunc]['Input Functions'].iteritems():
    #                     for ipf, sipfs in ipfs.iteritems():
    #                         for sipf in sipfs:
    #                             if param in funcWidget.input_functions:
    #                                 ifwidget = funcWidget.input_functions[param]
    #                             else:
    #                                 ifwidget = self.addInputFunction(funcWidget, param, ipf, sipf,
    #                                                                  package=reconpkg.packages[config_dict[sipf][1]])
    #                             if 'Enabled' in sipfs[sipf] and not sipfs[sipf]['Enabled']:
    #                                 ifwidget.enabled = False
    #                             if 'Parameters' in sipfs[sipf]:
    #                                 for p, v in sipfs[sipf]['Parameters'].iteritems():
    #                                     ifwidget.params.child(p).setValue(v)
    #                                     if setdefaults:
    #                                         ifwidget.params.child(p).setDefault(v)
    #                             ifwidget.updateParamsDict()
    #             funcWidget.updateParamsDict()
    #     self.sigPipelineChanged.emit()
