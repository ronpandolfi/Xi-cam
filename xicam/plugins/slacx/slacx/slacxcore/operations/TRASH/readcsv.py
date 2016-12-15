import numpy as np

from slacxop import Operation
import optools

#'/Users/Amanda/Desktop/Travails/Programming/ImageProcessing/SampleData/Liheng/SolventCorrection/R13.csv'
#'/Users/Amanda/Desktop/Travails/Programming/ImageProcessing/SampleData/Liheng/SolventCorrection/R4.csv'

class ReadCSV_13(Operation):
    """Read R13 file."""

    def __init__(self):
        input_names = []
        output_names = ['list_of_x_y_dy_bg_dbg_name']
        super(ReadCSV_13, self).__init__(input_names, output_names)
        self.output_doc['list_of_x_y_dy_bg_dbg_name'] = 'blank'
        self.categories = ['INPUT.APF TESTS']

    def run(self):
        filename = '/Users/Amanda/Desktop/Travails/Programming/ImageProcessing/SampleData/Liheng/SolventCorrection/R13.csv'
        cols = (0,1,2, 5,6, 9,10, 13,14, 17,18, 21,22)
        arr = np.loadtxt(filename, delimiter=',', skiprows=2, usecols=cols)
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


class ReadCSV_13_alt(Operation):
    """Read R13 file."""

    def __init__(self):
        input_names = []
        output_names = ['x', 'y', 'dy', 'bg', 'dbg']
        super(ReadCSV_13_alt, self).__init__(input_names, output_names)
        self.output_doc['x'] = 'blank'
        self.output_doc['y'] = 'blank'
        self.output_doc['dy'] = 'blank'
        self.output_doc['bg'] = 'blank'
        self.output_doc['dbg'] = 'blank'
        self.categories = ['INPUT.APF TESTS']

    def run(self):
        filename = '/Users/Amanda/Desktop/Travails/Programming/ImageProcessing/SampleData/Liheng/SolventCorrection/R13.csv'
        cols = (0,1,2, 5,6, 9,10, 13,14, 17,18, 21,22)
        arr = np.loadtxt(filename, delimiter=',', skiprows=2, usecols=cols)
        self.outputs['x'] = arr[:,0]
        self.outputs['bg'] = arr[:,1]
        self.outputs['dbg'] = arr[:,2]
        self.outputs['y'] = arr[:,7]
        self.outputs['dy'] = arr[:,8]


class ReadCSV_4_6(Operation):
    """Read R4 file."""

    def __init__(self):
        input_names = []
        output_names = ['list_of_x_y_dy_bg_dbg_name']
        super(ReadCSV_4_6, self).__init__(input_names, output_names)
        self.output_doc['list_of_x_y_dy_bg_dbg_name'] = 'blank'
        self.categories = ['INPUT.APF TESTS']

    def run(self):
        filename = '/Users/Amanda/Desktop/Travails/Programming/ImageProcessing/SampleData/Liheng/SolventCorrection/R4.csv'
        cols = (0,1,2, 5,6, 9,10, 13,14, 17,18, 21,22, 25,26, 29,30, 33,34)
        arr = np.loadtxt(filename, delimiter=',', skiprows=2, usecols=cols)
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
        self.input_doc['list1'] = 'blank'
        self.input_doc['list2'] = 'blank'
        self.output_doc['full_list'] = 'blank'
        # source & type
        self.input_src['list1'] = optools.wf_input
        self.input_src['list2'] = optools.wf_input
        self.categories = ['INPUT.APF TESTS']

    def run(self):
        self.outputs['full_list'] = []
        for ii in self.inputs['list1']:
            self.outputs['full_list'].append(ii)
        for ii in self.inputs['list2']:
            self.outputs['full_list'].append(ii)


class ReadMegaSAXS(Operation):
    """Read seven diffractograms from csv."""

    def __init__(self):
        input_names = []
        #output_names = ['q1','I1','dI1', 'q2','I2','dI2', 'q3','I3','dI3', 'q4','I4','dI4', 'q5','I5','dI5', 'q6','I6','dI6', 'q7','I7','dI7']
        output_names = ['list_of_x_y_dy']
        super(ReadMegaSAXS, self).__init__(input_names, output_names)
        self.output_doc['list_of_x_y_dy'] = 'blank'
        self.categories = ['INPUT.APF TESTS']

    def run(self):
        fileloc = '/Users/Amanda/Desktop/Travails/Programming/ImageProcessing/SampleData/Liheng/megaSAXSspreadsheet/megaSAXSspreadsheet.csv'
        data1 = np.loadtxt(fileloc, delimiter=',', skiprows=2, comments=',,,',
                           usecols=(0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26))
        data2 = np.loadtxt(fileloc, delimiter=',', skiprows=2, usecols=(24, 25, 26))
        self.outputs['list_of_x_y_dy'] = []
        self.outputs['list_of_x_y_dy'].append([data1[:,0], data1[:,1], data1[:,2]])
        self.outputs['list_of_x_y_dy'].append([data1[:,3], data1[:,4], data1[:,5]])
        self.outputs['list_of_x_y_dy'].append([data1[:,6], data1[:,7], data1[:,8]])
        self.outputs['list_of_x_y_dy'].append([data1[:,9], data1[:,10], data1[:,11]])
        self.outputs['list_of_x_y_dy'].append([data1[:,12], data1[:,13], data1[:,14]])
        self.outputs['list_of_x_y_dy'].append([data1[:,15], data1[:,16], data1[:,17]])
        self.outputs['list_of_x_y_dy'].append([data2[:,0], data2[:,1], data2[:,2]])
        #self.outputs['q1'] = data1[:,0]
        #self.outputs['I1'] = data1[:, 0]
        #self.outputs['dI1'] = data1[:, 0]
