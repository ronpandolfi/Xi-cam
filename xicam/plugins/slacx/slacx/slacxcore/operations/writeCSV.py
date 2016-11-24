from os.path import join, splitext, exists, isfile, split, isdir
from os import listdir, linesep, remove, getcwd
import numpy as np

from slacxop import Operation
import optools
from SSRL_1_5_readers import read_header


class WriteTemperatureIndex(Operation):
    """Find .csv diffractograms; match with temperatures from headers; record.

    If a temperature index already exists for some reason, it will be overwritten."""

    def __init__(self):
        input_names = ['directory']
        output_names = ['temperatures', 'filenames', 'temperature_index_file']
        super(WriteTemperatureIndex, self).__init__(input_names, output_names)
        # docstrings
        self.input_doc['directory'] = "path to directory with .csv's and .txt headers in it"
        self.output_doc['temperatures'] = 'temperatures from headers'
        self.output_doc['filenames'] = 'names of csvs'
        self.output_doc['temperature_index_file'] = 'csv-formatted file containing temperature indexed csv file names'
        # source & type
        self.input_src['directory'] = optools.fs_input
        self.categories = ['INPUT.MISC','OUTPUT.MISC']

    def run(self):
        directory = self.inputs['directory']
        outname = 'temperature_index.csv'
        outloc = join(directory,outname)
        try:
            remove(outloc)
        except:
            pass
        csvnames = find_csvs(directory)
        temperatures = []
        for ii in range(len(csvnames)):
            headernameii = txtname_from_csvname(csvnames[ii])
            headerii = read_header(headernameii)
            temp = headerii['temp_celsius']
            temperatures.append(temp)
        outfile = open(outloc, 'w')
        outfile.write("#TEMPCELSIUS,FILENAME")
        for ii in range(len(csvnames)):
            msg = str(temperatures[ii])+","+csvnames[ii]+linesep
            outfile.write(msg)
        outfile.close()
        self.outputs['filenames'] = csvnames
        self.outputs['temperatures'] = temperatures
        self.outputs['temperature_index_file'] = outloc

class ReadTemperatureIndex(Operation):
    """Read temperature index file written by WriteTemperatureIndex."""

    def __init__(self):
        input_names = ['directory']
        output_names = ['temperatures', 'filenames', 'temperature_index_file']
        super(ReadTemperatureIndex, self).__init__(input_names, output_names)
        self.input_doc['directory'] = "path to directory with background .csv's and .txt headers in it"
        self.output_doc['temperatures'] = 'temperatures from headers'
        self.output_doc['filenames'] = 'names of csvs'
        self.output_doc['temperature_index_file'] = 'csv-formatted file containing temperature indexed csv file names'
        self.categories = ['INPUT.MISC']

    def run(self):
        directory = self.inputs['directory']
        outname = 'temperature_index.csv'
        outloc = join(directory,outname)
        self.outputs['temperatures'] = np.loadtxt(outloc, dtype=float, delimiter=',', skiprows=1, usecols=(0))
        self.outputs['filenames'] = np.loadtxt(outloc, dtype=str, delimiter=',', skiprows=1, usecols=(1))
        self.outputs['temperature_index_file'] = outloc

class SelectClosestTemperatureBackgroundFromTemperature(Operation):
    """Read temperature index file written by WriteTemperatureIndex."""

    def __init__(self):
        input_names = ['directory','this_temperature']
        output_names = ['background_q','background_I']
        super(SelectClosestTemperatureBackgroundFromTemperature, self).__init__(input_names, output_names)
        # docstrings
        self.input_doc['directory'] = "path to directory with background .csv's and .txt headers in it"
        self.input_doc['this_temperature'] = "temperature we want to find a background for"
        self.output_doc['background_q'] = 'appropriate background q'
        self.output_doc['background_I'] = 'appropriate background I'
        # source & type
        self.input_src['directory'] = optools.fs_input
        self.input_src['this_temperature'] = optools.wf_input
        self.input_type['directory'] = optools.str_type
        self.input_type['this_temperature'] = optools.float_type
        self.categories = ['1D DATA PROCESSING.BACKGROUND SUBTRACTION']

    def run(self):
        directory = self.inputs['directory']
        this_temperature = self.inputs['this_temperature']
        indexname = 'temperature_index.csv'
        indexloc = join(directory,indexname)
        temperatures = np.loadtxt(indexloc, dtype=float, delimiter=',', skiprows=1, usecols=(0))
        filenames = np.loadtxt(indexloc, dtype=str, delimiter=',', skiprows=1, usecols=(1))
        diff = np.fabs(temperatures - this_temperature)
        index_of_best_temp = np.where(diff == diff.min())[0][0]
        file_of_best_temp = filenames[index_of_best_temp]
        q, I = read_csv_q_I(file_of_best_temp)
        self.outputs['background_q'] = q
        self.outputs['background_I'] = I


class SelectClosestTemperatureBackgroundFromHeader(Operation):
    """Read temperature index file written by WriteTemperatureIndex."""

    def __init__(self):
        input_names = ['directory','this_header']
        output_names = ['background_q','background_I']
        super(SelectClosestTemperatureBackgroundFromHeader, self).__init__(input_names, output_names)
        # docstrings
        self.input_doc['directory'] = "path to directory with background .csv's and .txt headers in it"
        self.input_doc['this_header'] = "header of this data; will use entry *temp_celsius*"
        self.output_doc['background_q'] = 'appropriate background q'
        self.output_doc['background_I'] = 'appropriate background I'
        # source & type
        self.input_src['directory'] = optools.fs_input
        self.input_src['this_header'] = optools.wf_input
        self.input_type['directory'] = optools.str_type
        self.categories = ['1D DATA PROCESSING.BACKGROUND SUBTRACTION']

    def run(self):
        print "SelectClosestTemperatureBackgroundFromHeader running"
        print "directory is", self.inputs['directory']
        directory = self.inputs['directory']
        print "temperature is", self.inputs['this_header']['temp_celsius']
        this_temperature = self.inputs['this_header']['temp_celsius']
        indexname = 'temperature_index.csv'
        indexloc = join(directory,indexname)
        print "looking for index file at %s" % indexloc
        temperatures = np.loadtxt(indexloc, dtype=float, delimiter=',', skiprows=1, usecols=(0,))
        print "temperatures read"
        filenames = np.loadtxt(indexloc, dtype=str, delimiter=',', skiprows=1, usecols=(1,))
        print "file names read"
        diff = np.fabs(temperatures - this_temperature)
        index_of_best_temp = np.where(diff == diff.min())[0][0]
        file_of_best_temp = filenames[index_of_best_temp]
        print "background file %s selected" % file_of_best_temp
        q, I = read_csv_q_I(file_of_best_temp)
        print "background q, I read from %s" % file_of_best_temp
        print "q.shape, I.shape, q.dtype, I.dtype", q.shape, I.shape, q.dtype, I.dtype
        self.outputs['background_q'] = q
        self.outputs['background_I'] = I


class WriteCSV_q_I(Operation):
    """Write q and I to a csv-formatted file."""

    def __init__(self):
        input_names = ['q','I','image_location']
        output_names = ['csv_location']
        super(WriteCSV_q_I, self).__init__(input_names, output_names)
        # docstrings
        self.input_doc['q'] = "1d ndarray; independent variable"
        self.input_doc['I'] = "1d ndarray; dependent variable; same shape as *q*"
        self.input_doc['image_location'] = "string path to "

        self.input_doc['this_temperature'] = "temperature we want to find a background for"
        self.output_doc['background_q'] = 'appropriate background q'
        self.output_doc['background_I'] = 'appropriate background I'
        # source & type
        self.input_src['q'] = optools.wf_input
        self.input_src['I'] = optools.wf_input
        self.input_src['image_location'] = optools.wf_input
        self.input_type['image_location'] = optools.str_type
        self.categories = ['OUTPUT']

    def run(self):
        csv_location = csvname_from_imagename(self.inputs['image_location'])
        write_csv_q_I(self.inputs['q'], self.inputs['I'], csv_location)
        self.outputs['csv_location'] = csv_location

class WriteCSV_q_I_dI(Operation):
    """Write q, I, and dI to a csv-formatted file."""

    def __init__(self):
        input_names = ['q','I','dI','image_location']
        output_names = ['csv_location']
        super(WriteCSV_q_I_dI, self).__init__(input_names, output_names)
        # source & type
        self.input_src['q'] = optools.wf_input
        self.input_src['I'] = optools.wf_input
        self.input_src['dI'] = optools.wf_input
        self.input_src['image_location'] = optools.wf_input
        self.input_type['image_location'] = optools.str_type
        self.categories = ['OUTPUT']

    def run(self):
        csv_location = csvname_from_imagename(self.inputs['image_location'])
        write_csv_q_I_dI(self.inputs['q'], self.inputs['I'], self.inputs['dI'], csv_location)
        self.outputs['csv_location'] = csv_location


def write_csv_q_I(q, I, nameloc):
    datablock = np.zeros((q.size,2),dtype=float)
    datablock[:,0] = q
    datablock[:,1] = I
    np.savetxt(nameloc, datablock, delimiter=',', newline=linesep, header='q, I')

def write_csv_q_I_dI(q, I, dI, nameloc):
    datablock = np.zeros((q.size,3),dtype=float)
    datablock[:,0] = q
    datablock[:,1] = I
    datablock[:,2] = dI
    np.savetxt(nameloc, datablock, delimiter=',', newline=linesep, header='q, I')


def find_csvs(directory):
    innames = listdir(directory)
    csvnames = []
    for ii in range(len(innames)):
        innameii = splitext(innames[ii])
        if innameii[1] == '.csv':
            csvnames.append(join(directory, innames[ii]))
    return csvnames

def txtname_from_csvname(csvname):
    root = splitext(csvname)[0]
    headername = root + '.txt'
    return headername

def csvname_from_imagename(imagename):
    root = splitext(imagename)[0]
    csvname = root + '.csv'
    return csvname


def read_csv_q_I(nameloc):
    directory = split(nameloc)[0]
    data = np.loadtxt(nameloc, dtype=float, delimiter=',', skiprows=1, usecols=(0,1))
    q = data[:,0]
    I = data[:,1]
    return q, I

def read_csv_q_I_dI(nameloc):
    data = np.loadtxt(nameloc, dtype=float, delimiter=',', skiprows=1, usecols=(0,1,2))
    q = data[:,0]
    I = data[:,1]
    dI = data[:,2]
    return q, I, dI

'''
from os.path import exists, isfile, split, isdir
nameloc = "/Users/Amanda/Data20161118/ODE/ODE_SAXS1_0002.csv"
nameloc = "/Users/Amanda/Data20161118/ODE/ODE_SAXS1_0002.csv"
import numpy as np
data = np.loadtxt(nameloc, dtype=float, delimiter=',', skiprows=1, usecols=(0,1))
data1 = np.loadtxt(nameloc, dtype=float, delimiter=',', skiprows=1, usecols=(0,1))
data2 = np.loadtxt(nameloc, dtype=float, delimiter=',', skiprows=1, usecols=(0,1))
print "file exists?: ", exists(nameloc)

(data1 == data).all()
(data1 == data2).all()
'''