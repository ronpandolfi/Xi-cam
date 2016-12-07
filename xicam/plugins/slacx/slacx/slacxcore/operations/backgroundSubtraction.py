import numpy as np

from slacxop import Operation
import optools

class BgSubtractByTemperature(Operation):
    """
    Find a background spectrum from a batch of background spectra,
    where the temperature of the background spectrum is as close as possible
    to the temperature of the measured spectrum.
    Then subtract that background spectrum from the input spectrum.
    """
    
    def __init__(self):
        input_names = ['q_I_meas', 'header', 'bg_batch_output']
        output_names = ['q_I_bgsub', 'bg_factor']
        super(BgSubtractByTemperature, self).__init__(input_names, output_names)
        self.input_doc['q_I_meas'] = 'windowed n-by-2 array of q and I(q)'
        self.input_doc['header'] = 'header file generated with the measured spectrum, expected to be a dict with key \'TEMP\''
        self.input_doc['bg_batch_output'] = 'the output (list of dicts) of a batch of background spectra at different temperatures'
        self.output_doc['q_I_bgsub'] = 'q_I_meas - bg_factor * (q_I_bg)'
        self.output_doc['bg_factor'] = 'correction factor applied to background before subtraction to ensure positive intensity values'
        self.input_src['q_I_meas'] = optools.wf_input
        self.input_src['header'] = optools.wf_input
        self.input_src['bg_batch_output'] = optools.wf_input
        self.categories = ['1D DATA PROCESSING.BACKGROUND SUBTRACTION']

    def run(self):
        q_I = self.inputs['q_I_meas']
        T = self.inputs['header']['TEMP']
        #print 'T is {}'.format(T)
        bg_out = self.inputs['bg_batch_output']
        T_allbg = [d['header']['TEMP'] for d in bg_out]
        #print 'bg T values are {}'.format(T_allbg)
        q_I_allbg = [d['q_I_window'] for d in bg_out]
        idx = np.argmin(np.abs([T_bg - T for T_bg in T_allbg]))
        #print 'idx of closest T is {}'.format(idx)
        q_I_bg = q_I_allbg[idx]
        if not all(q_I[:,0] == q_I_bg[:,0]):
            msg = 'SPECTRUM AND BACKGROUND ON DIFFERENT q DOMAINS'
            raise ValueError(msg)
        bad_data = (q_I[:,1] <= 0) | (q_I_bg[:,1] <= 0) | np.isnan(q_I[:,1]) | np.isnan(q_I_bg[:,1])
        bg_factor = np.min(q_I[:,1][~bad_data] / q_I_bg[:,1][~bad_data])
        #print 'bg factor is {}'.format(bg_factor)
        I_bgsub = q_I[:,1] - bg_factor * q_I_bg[:,1]
        self.outputs['q_I_bgsub'] = np.array(zip(q_I[:,0],I_bgsub))
        self.outputs['bg_factor'] = bg_factor

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
    bad_data = (foreground <= 0) | (background <= 0) | np.isnan(foreground) | np.isnan(background)
    if np.any(bad_data):
        print "There were %i invalid data points in this background subtraction attempt." % np.sum(bad_data)
    factor = np.min(foreground[~bad_data] / background[~bad_data])
    #factor = np.min(foreground / background)
    if (factor > 1) or (factor < 0.8):
        print "The background multiplication factor was %f, an unusual value." % factor
    subtracted = foreground - (factor * background)
    return subtracted, factor

def subtract_maximum_background_with_errors(foreground, background, foreground_error, background_error):
    bad_data = (foreground < 0) | (background < 0) | np.isnan(foreground) | np.isnan(background)
    if np.any(bad_data):
        print "There were %i invalid data points in this background subtraction attempt." % np.sum(bad_data)
    factor = np.min(foreground[~bad_data] / background[~bad_data])
    #factor = np.min(foreground / background)
    if (factor > 1) or (factor < 0.8):
        print "The background multiplication factor was %f, an unusual value." % factor
    subtracted = foreground - (factor * background)
    subtracted_error = (foreground_error**2 + (factor * background_error)**2)**0.5
    return subtracted, subtracted_error, factor








