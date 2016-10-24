from core.operations.slacxop import Operation

import numpy as np

#'/Users/Amanda/Desktop/Travails/Programming/ImageProcessing/SampleData/Liheng/SolventCorrection/R13.csv'
#'/Users/Amanda/Desktop/Travails/Programming/ImageProcessing/SampleData/Liheng/SolventCorrection/R4.csv'

class ReadCSV1(Operation):
    """Read R13 file."""

    def __init__(self):
        input_names = ['file']
        output_names = ['list_of_x_y_dy_bg_dbg_name']
        super(ReadCSV1, self).__init__(input_names, output_names)
        self.input_doc['file'] = ''
        self.output_doc['list_of_x_y_dy_bg_dbg_name'] = 'blank'

    def run(self):
        cols = (0,1,2, 5,6, 9,10, 13,14, 17,18, 21,22)
        arr = np.loadtxt(self.inputs['file'], delimiter=',', skiprows=2, usecols=cols)
        x = arr[:,0]
        bg1 = arr[:,1]
        dbg1 = arr[:,2]
        bg2 = arr[:,3]
        dbg2 = arr[:,4]
        bg3 = arr[:,5]
        dbg3 = arr[:,6]
        y1 = arr[:,7]
        dy1 = arr[:,8]
        y2 = arr[:,9]
        dy2 = arr[:,10]
        y3 = arr[:,11]
        dy3 = arr[:,12]
        self.outputs['list_of_x_y_dy_bg_dbg_name'] = []
        self.outputs['list_of_x_y_dy_bg_dbg_name'].append((x, y1, dy1, bg1, dbg1, 'R13 30'))
        self.outputs['list_of_x_y_dy_bg_dbg_name'].append((x, y2, dy2, bg2, dbg2, 'R13 40'))
        self.outputs['list_of_x_y_dy_bg_dbg_name'].append((x, y3, dy3, bg3, dbg3, 'R13 50'))


class ReadCSV2(Operation):
    """Read R4 file."""

    def __init__(self):
        input_names = ['file']
        output_names = ['list_of_x_y_dy_bg_dbg_name']
        super(ReadCSV2, self).__init__(input_names, output_names)
        self.input_doc['file'] = ''
        self.output_doc['list_of_x_y_dy_bg_dbg_name'] = 'blank'

    def run(self):
        cols = (0,1,2, 5,6, 9,10, 13,14, 17,18, 21,22, 25,26, 29,30, 33,34)
        arr = np.loadtxt(self.inputs['file'], delimiter=',', skiprows=2, usecols=cols)
        x = arr[:, 0]
        y1 = arr[:, 1]
        dy1 = arr[:, 2]
        y2 = arr[:, 3]
        dy2 = arr[:, 4]
        y3 = arr[:, 5]
        dy3 = arr[:, 6]
        bg1 = arr[:, 7]
        dbg1 = arr[:, 8]
        bg2 = arr[:, 9]
        dbg2 = arr[:, 10]
        bg3 = arr[:, 11]
        dbg3 = arr[:, 12]
        y4 = arr[:, 13]
        dy4 = arr[:, 14]
        y5 = arr[:, 15]
        dy5 = arr[:, 16]
        y6 = arr[:, 17]
        dy6 = arr[:, 18]
        bg4 = bg1
        dbg4 = dbg1
        bg5 = bg2
        dbg5 = dbg2
        bg6 = bg3
        dbg6 = dbg3
        self.outputs['list_of_x_y_dy_bg_dbg_name'] = []
        self.outputs['list_of_x_y_dy_bg_dbg_name'].append((x, y1, dy1, bg1, dbg1, 'R4 30'))
        self.outputs['list_of_x_y_dy_bg_dbg_name'].append((x, y2, dy2, bg2, dbg2, 'R4 40'))
        self.outputs['list_of_x_y_dy_bg_dbg_name'].append((x, y3, dy3, bg3, dbg3, 'R4 50'))
        self.outputs['list_of_x_y_dy_bg_dbg_name'].append((x, y4, dy4, bg4, dbg4, 'R6 30'))
        self.outputs['list_of_x_y_dy_bg_dbg_name'].append((x, y5, dy5, bg5, dbg5, 'R6 40'))
        self.outputs['list_of_x_y_dy_bg_dbg_name'].append((x, y6, dy6, bg6, dbg6, 'R6 50'))

class concatCSV1CSV2(Operation):
    """Smash CSV1 and CSV2 outputs into a single list."""

    def __init__(self):
        input_names = ['list1', 'list2']
        output_names = ['full_list']
        super(concatCSV1CSV2, self).__init__(input_names, output_names)
        self.input_doc['list1'] = ''
        self.input_doc['list2'] = ''
        self.output_doc['full_list'] = 'blank'

    def run(self):
        self.outputs['full_list'] = []
        for ii in self.inputs['list1']:
            self.outputs['full_list'].append(ii)
        for ii in self.inputs['list2']:
            self.outputs['full_list'].append(ii)