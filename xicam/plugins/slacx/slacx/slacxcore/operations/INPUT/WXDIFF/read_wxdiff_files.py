from ...slacxop import Operation
from ... import optools

class WXDCalibToDict(Operation):
    """
    Input is the path to a WXDiff .calib file 
    output is a dict containing calib parameters 
    """
    def __init__(self):
        input_names = ['calib_file']
        output_names = ['calib_dict']
        super(WXDCalibToDict,self).__init__(input_names,output_names)
        self.input_doc['calib_file'] = 'filesystem path to a .calib file as produced by WXDiff calibration'
        self.output_doc['calib_dict'] = 'a dict containing the calibration parameters from the file'
        self.input_src['calib_file'] = optools.fs_input
        self.categories = ['INPUT.WXDIFF']

    def run(self):
        d = {}
        for line in open(self.inputs['calib_file'],'r'):
            kv = line.strip().split('=')
            if kv[0] == 'bcenter_x':
                d['x0_pixel'] = float(kv[1])
            if kv[0] == 'bcenter_y':
                d['y0_pixel'] = float(kv[1])
            if kv[0] == 'detect_dist':
                d['d_pixel'] = float(kv[1])
            if kv[0] == 'detect_tilt_alpha':
                d['rotation_rad'] = float(kv[1])
            if kv[0] == 'detect_tilt_delta':
                d['tilt_rad'] = float(kv[1])
            if kv[0] == 'wavelenght':
                d['lambda'] = float(kv[1])
        self.outputs['calib_dict'] = d


