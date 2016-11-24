import numpy as np

from slacxop import Operation
import optools



class SubtractMaximumBackgroundNoErrors(Operation):
    """Subtract a background from a foreground, with scaling to prevent over-subtraction."""
    def __init__(self):
        input_names = ['foreground', 'background']
        output_names = ['subtracted', 'factor']
        super(SubtractMaximumBackgroundNoErrors, self).__init__(input_names, output_names)
        self.input_doc['foreground'] = '1d ndarray; experimental data'
        self.input_doc['background'] = '1d ndarray; background to subtract, same coordinates as *foreground*'
        self.output_doc['subtracted'] = 'background-subtracted experimental data'
        self.output_doc['factor'] = 'the factor the background was multiplied by before subraction'
        # source & type
        self.input_src['foreground'] = optools.wf_input
        self.input_src['background'] = optools.wf_input
        self.categories = ['1D DATA PROCESSING.BACKGROUND SUBTRACTION']

    def run(self):
        self.outputs['subtracted'], self.outputs['factor'] = subtract_maximum_background_no_errors(
            self.inputs['foreground'], self.inputs['background'])

class SubtractMaximumBackgroundWithErrors(Operation):
    """Subtract a background from a foreground, with scaling to prevent over-subtraction."""
    def __init__(self):
        input_names = ['foreground', 'background', 'foreground_error', 'background_error']
        output_names = ['subtracted', 'subtracted_error', 'factor']
        super(SubtractMaximumBackgroundWithErrors, self).__init__(input_names, output_names)
        self.input_doc['foreground'] = '1d ndarray; experimental data'
        self.input_doc['background'] = '1d ndarray; background to subtract, same coordinates as *foreground*'
        self.input_doc['foreground_error'] = '1d ndarray; error estimate of *foreground*'
        self.input_doc['background_error'] = '1d ndarray; error estimate of *background*'
        self.output_doc['subtracted'] = 'background-subtracted experimental data'
        self.output_doc['subtracted_error'] = 'error estimate of *subtracted*'
        self.output_doc['factor'] = 'the factor the background was multiplied by before subraction'
        # source & type
        self.input_src['foreground'] = optools.wf_input
        self.input_src['background'] = optools.wf_input
        self.input_src['foreground_error'] = optools.wf_input
        self.input_src['background_error'] = optools.wf_input
        self.categories = ['1D DATA PROCESSING.BACKGROUND SUBTRACTION']

    def run(self):
        self.outputs['subtracted'], self.outputs['subtracted_error'], self.outputs['factor'] = \
            subtract_maximum_background_with_errors(self.inputs['foreground'], self.inputs['background'],
                                                    self.inputs['foreground_error'], self.inputs['background_error'])

def subtract_maximum_background_no_errors(foreground, background):
    print "NaNs in foreground, background?", np.any(np.isnan(foreground)), np.any(np.isnan(background))
    factor = np.min(foreground / background)
    subtracted = foreground - (factor * background)
    return subtracted, factor

def subtract_maximum_background_with_errors(foreground, background, foreground_error, background_error):
    factor = np.min(foreground / background)
    subtracted = foreground - (factor * background)
    subtracted_error = (foreground_error**2 + (factor * background_error)**2)**0.5
    return subtracted, subtracted_error, factor

