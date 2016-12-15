import numpy as np
import datetime
import time

import pytz
import tzlocal

from ..slacxop import Operation
from .. import optools

class WindowZip(Operation):
    """
    From input iterables of x and y, 
    produce an n-by-2 array 
    where x is bounded by the specified limits 
    """

    def __init__(self):
        input_names = ['x','y','x_min','x_max']
        output_names = ['x_y_window']
        super(WindowZip,self).__init__(input_names,output_names)        
        self.input_src['x'] = optools.wf_input
        self.input_src['y'] = optools.wf_input
        self.input_src['x_min'] = optools.user_input
        self.input_src['x_max'] = optools.user_input
        self.input_type['x_min'] = optools.float_type
        self.input_type['x_max'] = optools.float_type
        self.inputs['x_min'] = 0.02 
        self.inputs['x_max'] = 0.6 
        self.input_doc['x'] = 'list (or iterable) of x values'
        self.input_doc['y'] = 'list (or iterable) of y values'
        self.output_doc['x_y_window'] = 'n-by-2 array with x, y pairs for x_min<x<x_max'
        self.categories = ['PACKAGING']

    def run(self):
        xvals = self.inputs['x']
        yvals = self.inputs['y']
        x_min = self.inputs['x_min']
        x_max = self.inputs['x_max']
        idx_good = ((xvals > x_min) & (xvals < x_max))
        x_y_window = np.zeros((idx_good.sum(),2))
        x_y_window[:,0] = xvals[idx_good]
        x_y_window[:,1] = yvals[idx_good]
        self.outputs['x_y_window'] = x_y_window

class TimeTempFromHeader(Operation):
    """
    Get time and temperature from a detector output header file.
    Return string time, float time (utc in seconds), and float temperature.
    Time is assumed to be in the format Day Mon dd hh:mm:ss yyyy.
    """
    def __init__(self):
        input_names = ['header_dict','time_key','temp_key']
        output_names = ['time_str','time','temp']
        super(TimeTempFromHeader,self).__init__(input_names,output_names)        
        self.input_src['header_dict'] = optools.wf_input
        self.input_src['time_key'] = optools.user_input
        self.input_src['temp_key'] = optools.user_input
        self.input_type['time_key'] = optools.str_type
        self.input_type['temp_key'] = optools.str_type
        self.inputs['time_key'] = 'time'
        self.inputs['temp_key'] = 'TEMP'
        self.input_doc['header_dict'] = 'workflow uri of dict produced from detector output header file.'
        self.input_doc['time_key'] = 'key in header_dict that refers to the time' 
        self.input_doc['temp_key'] = 'key in header_dict that refers to the temperature' 
        self.output_doc['time_str'] = 'string representation of the time'
        self.output_doc['time'] = 'UTC time in seconds'
        self.output_doc['temp'] = 'Temperature'
        self.categories = ['PACKAGING']

    def run(self):
        d = self.inputs['header_dict']
        time_str = str(d[self.inputs['time_key']])
        temp = float(d[self.inputs['temp_key']])
        # process the UTC time in seconds assuming %a %b %d %H:%M:%S %Y format
        # set local time zone for utc-awareness 
        tz = tzlocal.get_localzone()
        # use strptime to create a naive datetime object
        dt = datetime.datetime.strptime(time_str.strip(),"%a %b %d %H:%M:%S %Y")
        # add in timezone information to make a utc-aware datetime object
        dt_aware = datetime.datetime(dt.year,dt.month,dt.day,dt.hour,dt.minute,dt.second,dt.microsecond,tz)
        # interpret the time in UTC milliseconds
        t_utc = time.mktime(dt_aware.timetuple())
        self.outputs['time_str'] = time_str
        self.outputs['time'] = float(t_utc)
        self.outputs['temp'] = temp

class XYDataFromBatch(Operation):
    """
    From input iterables of x and y, 
    produce an n-by-2 array 
    """

    def __init__(self):
        input_names = ['batch_output','x_key','y_key','x_shift_flag']
        output_names = ['x_y']
        super(XYDataFromBatch,self).__init__(input_names,output_names)        
        self.input_src['batch_output'] = optools.wf_input
        self.input_src['x_key'] = optools.user_input
        self.input_src['y_key'] = optools.user_input
        self.input_src['x_shift_flag'] = optools.user_input
        self.input_type['x_key'] = optools.str_type
        self.input_type['y_key'] = optools.str_type
        self.input_type['x_shift_flag'] = optools.bool_type
        self.inputs['x_key'] = 'Operation.outputs.name'
        self.inputs['y_key'] = 'Operation.outputs.name'
        self.inputs['x_shift_flag'] = False
        self.input_doc['batch_output'] = 'list of dicts produced by a batch execution. keyed by workflow uris.'
        self.input_doc['x_key'] = 'uri of data for x. Must be in batch.saved_items(). User input as a string.'
        self.input_doc['y_key'] = 'uri of data for y. Must be in batch.saved_items(). User input as a string.'
        self.input_doc['x_shift_flag'] = 'if True, shift x data so that its minimum value is zero.' 
        self.output_doc['x_y'] = 'n-by-2 array of x and y values, sorted on the x values, shifted to x=0.'
        self.categories = ['PACKAGING']

    def run(self):
        b_out = self.inputs['batch_output']
        x_key = self.inputs['x_key']
        y_key = self.inputs['y_key']
        x_all = np.array([d[x_key] for d in b_out],dtype=float)
        if self.inputs['x_shift_flag']:
            x_all = x_all - np.min(x_all)
        y_all = np.array([d[y_key] for d in b_out],dtype=float)
        xmin = np.min(x_all)
        #import pdb; pdb.set_trace()
        self.outputs['x_y'] = np.sort(np.array(zip(x_all,y_all)),0)

class Window_q_I_2(Operation):
    """
    From input iterables of *q_list_in* and *I_list_in*,
    produce two 1d vectors
    where q is greater than *q_min* and less than *q_min*
    """

    def __init__(self):
        input_names = ['q_list_in','I_list_in','q_min','q_max']
        output_names = ['q_list_out','I_list_out']
        super(Window_q_I_2,self).__init__(input_names,output_names)
        # docstrings
        self.input_doc['q_list_in'] = '1d iterable listing q values'
        self.input_doc['I_list_in'] = '1d iterable listing I values'
        self.input_doc['q_min'] = 'lowest value of q of interest'
        self.input_doc['q_max'] = 'highest value of q of interest'
        self.output_doc['q_list_out'] = '1d iterable listing q values where q is between *q_min* and *q_max*'
        self.output_doc['I_list_out'] = '1d iterable listing I values where q is between *q_min* and *q_max*'
        # source & type
        self.input_src['q_list_in'] = optools.wf_input
        self.input_src['I_list_in'] = optools.wf_input
        self.input_src['q_min'] = optools.user_input
        self.input_src['q_max'] = optools.user_input
        self.input_type['q_min'] = optools.float_type
        self.input_type['q_max'] = optools.float_type
        self.inputs['q_min'] = 0.02
        self.inputs['q_max'] = 0.6
        self.categories = ['PACKAGING']

    def run(self):
        qvals = self.inputs['q_list_in']
        ivals = self.inputs['I_list_in']
        q_min = self.inputs['q_min']
        q_max = self.inputs['q_max']
        if (q_min >= q_max):
            raise ValueError("*q_max* must be greater than *q_min*.")
        good_qvals = ((qvals > q_min) & (qvals < q_max))
        self.outputs['q_list_out'] = qvals[good_qvals]
        self.outputs['I_list_out'] = ivals[good_qvals]


