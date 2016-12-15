from os.path import join, splitext, exists, isfile, split, isdir
from os import listdir, linesep, remove, getcwd
import numpy as np

from ..slacxop import Operation
from ..import optools

class WriteTemperatureIndex(Operation):
    """Find .csv diffractograms; match with temperatures from headers; record.

    If a temperature index already exists for some reason, it will be overwritten."""

    def __init__(self):
        input_names = ['background_directory']
        output_names = ['temperatures', 'filenames', 'temperature_index_file']
        super(WriteTemperatureIndex, self).__init__(input_names, output_names)
        # docstrings
        self.input_doc['background_directory'] = "path to directory with .csv's and .txt headers in it"
        self.output_doc['temperatures'] = 'temperatures from headers'
        self.output_doc['filenames'] = 'names of csvs'
        self.output_doc['temperature_index_file'] = 'csv-formatted file containing temperature indexed csv file names'
        # source & type
        self.input_src['directory'] = optools.fs_input
        self.categories = ['INPUT.MISC','OUTPUT.MISC']

    def run(self):
        directory = self.inputs['background_directory']
        outname = 'temperature_index.csv'
        outloc = join(directory,outname)
        try:
            remove(outloc)
        except:
            pass
        csvnames = find_by_extension(directory, '.csv')
        temperatures = []
        for ii in range(len(csvnames)):
            headernameii = replace_extension(csvnames[ii],'.txt')
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
        input_names = ['background_directory']
        output_names = ['temperatures', 'filenames', 'temperature_index_file']
        super(ReadTemperatureIndex, self).__init__(input_names, output_names)
        self.input_doc['background_directory'] = "path to directory with background .csv's and .txt headers in it"
        self.output_doc['temperatures'] = 'temperatures from headers'
        self.output_doc['filenames'] = 'names of csvs'
        self.output_doc['temperature_index_file'] = 'csv-formatted file containing temperature indexed csv file names'
        self.categories = ['INPUT.MISC']

    def run(self):
        directory = self.inputs['background_directory']
        outname = 'temperature_index.csv'
        outloc = join(directory,outname)
        self.outputs['temperatures'] = np.loadtxt(outloc, dtype=float, delimiter=',', skiprows=1, usecols=(0,))
        self.outputs['filenames'] = np.loadtxt(outloc, dtype=str, delimiter=',', skiprows=1, usecols=(1,))
        self.outputs['temperature_index_file'] = outloc

class SelectClosestTemperatureBackgroundFromTemperature(Operation):
    """Read temperature index file written by WriteTemperatureIndex."""

    def __init__(self):
        input_names = ['background_directory','this_temperature']
        output_names = ['background_q','background_I']
        super(SelectClosestTemperatureBackgroundFromTemperature, self).__init__(input_names, output_names)
        # docstrings
        self.input_doc['background_directory'] = "path to directory with background .csv's and .txt headers in it"
        self.input_doc['this_temperature'] = "temperature we want to find a background for"
        self.output_doc['background_q'] = 'appropriate background q'
        self.output_doc['background_I'] = 'appropriate background I'
        # source & type
        self.input_src['background_directory'] = optools.fs_input
        self.input_src['this_temperature'] = optools.wf_input
        self.input_type['background_directory'] = optools.str_type
        self.input_type['this_temperature'] = optools.float_type
        self.categories = ['1D DATA PROCESSING.BACKGROUND SUBTRACTION']

    def run(self):
        directory = self.inputs['background_directory']
        this_temperature = self.inputs['this_temperature']
        indexname = 'temperature_index.csv'
        indexloc = join(directory,indexname)
        temperatures = np.loadtxt(indexloc, dtype=float, delimiter=',', skiprows=1, usecols=(0,))
        filenames = np.loadtxt(indexloc, dtype=str, delimiter=',', skiprows=1, usecols=(1,))
        diff = np.fabs(temperatures - this_temperature)
        index_of_best_temp = np.where(diff == diff.min())[0][0]
        file_of_best_temp = filenames[index_of_best_temp]
        q, I = read_csv_q_I(file_of_best_temp)
        self.outputs['background_q'] = q
        self.outputs['background_I'] = I



class SelectClosestTemperatureBackgroundFromHeader2(Operation):
    """Read temperature index file written by WriteTemperatureIndex."""

    def __init__(self):
        input_names = ['temperatures','filenames','header']
        output_names = ['background_tiffile']
        super(SelectClosestTemperatureBackgroundFromHeader2, self).__init__(input_names, output_names)
        # docstrings
        self.input_doc['temperatures'] = "list or array of temperatures"
        self.input_doc['filenames'] = "names of tif files"
        self.input_doc['this_temperature'] = "temperature we want to find a background for"
        self.output_doc['background_tiffile'] = 'appropriate background tif file image'
        # source & type
        self.input_src['temperatures'] = optools.wf_input
        self.input_src['filenames'] = optools.wf_input
        self.input_src['header'] = optools.wf_input
        self.categories = ['1D DATA PROCESSING.BACKGROUND SUBTRACTION']

    def run(self):
        this_temperature = self.inputs['header']['temp_celsius']
        temperatures = np.array(self.inputs['temperatures'])
        filenames = self.inputs['filenames']
        diff = np.fabs(temperatures - this_temperature)
        index_of_best_temp = np.where(diff == diff.min())[0][0]
        file_of_best_temp = filenames[index_of_best_temp]
        self.outputs['background_tiffile'] = file_of_best_temp


class ConstructTemperatureIndex(Operation):
    """Find .csv diffractograms; match with temperatures from headers; record.

    If a temperature index already exists for some reason, it will be overwritten."""

    def __init__(self):
        input_names = ['background_directory']
        output_names = ['temperatures', 'filenames']
        super(ConstructTemperatureIndex, self).__init__(input_names, output_names)
        # docstrings
        self.input_doc['background_directory'] = "path to directory with .tif's and .txt headers in it"
        self.output_doc['temperatures'] = 'temperatures from headers'
        self.output_doc['filenames'] = 'names of csvs'
        #self.output_doc['temperature_index_file'] = 'csv-formatted file containing temperature indexed csv file names'
        # source & type
        self.input_src['directory'] = optools.fs_input
        self.categories = ['INPUT.MISC']

    def run(self):
        directory = self.inputs['background_directory']
        tifnames, temperatures = find_background_temperatures(directory)
        self.outputs['filenames'] = tifnames
        self.outputs['temperatures'] = temperatures



def find_background_temperatures(directory):
    tifnames = find_by_extension(directory, '.tif')
    txtnames = find_by_extension(directory, '.txt')
    temperatures = []
    for ii in range(len(txtnames)):
        headerii = read_header(txtnames[ii])
        temp = headerii['temp_celsius']
        temperatures.append(temp)
    return tifnames, temperatures

class SelectClosestTemperatureBackgroundFromHeader(Operation):
    """Read temperature index file written by WriteTemperatureIndex."""

    def __init__(self):
        input_names = ['background_directory','this_header']
        output_names = ['background_q','background_I']
        super(SelectClosestTemperatureBackgroundFromHeader, self).__init__(input_names, output_names)
        # docstrings
        self.input_doc['background_directory'] = "path to directory with background .csv's and .txt headers in it"
        self.input_doc['this_header'] = "header of this data; will use entry *temp_celsius*"
        self.output_doc['background_q'] = 'appropriate background q'
        self.output_doc['background_I'] = 'appropriate background I'
        # source & type
        self.input_src['background_directory'] = optools.fs_input
        self.input_src['this_header'] = optools.wf_input
        self.input_type['background_directory'] = optools.str_type
        self.categories = ['1D DATA PROCESSING.BACKGROUND SUBTRACTION']

    def run(self):
        directory = self.inputs['background_directory']
        this_temperature = self.inputs['this_header']['temp_celsius']
        indexname = 'temperature_index.csv'
        indexloc = join(directory,indexname)
        temperatures = np.loadtxt(indexloc, dtype=float, delimiter=',', skiprows=1, usecols=(0,))
        filenames = np.loadtxt(indexloc, dtype=str, delimiter=',', skiprows=1, usecols=(1,))
        diff = np.fabs(temperatures - this_temperature)
        index_of_best_temp = np.where(diff == diff.min())[0][0]
        file_of_best_temp = filenames[index_of_best_temp]
        q, I = read_csv_q_I(file_of_best_temp)
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
        # source & type
        self.input_src['q'] = optools.wf_input
        self.input_src['I'] = optools.wf_input
        self.input_src['image_location'] = optools.wf_input
        self.input_type['image_location'] = optools.str_type
        self.categories = ['OUTPUT.CSV']

    def run(self):
        csv_location = replace_extension(self.inputs['image_location'], '.csv')
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
        self.categories = ['OUTPUT.CSV']

    def run(self):
        csv_location = replace_extension(self.inputs['image_location'], '.csv')
        write_csv_q_I_dI(self.inputs['q'], self.inputs['I'], self.inputs['dI'], csv_location)
        self.outputs['csv_location'] = csv_location


class ReadCSV_q_I_dI(Operation):
    """Read q, I, and (if available) dI from a csv-formatted file.

    If the csv has no third column, returns a length-one array of zeros for dI."""

    def __init__(self):
        input_names = ['csv_location']
        output_names = ['q','I', 'dI']
        super(ReadCSV_q_I_dI, self).__init__(input_names, output_names)
        # docstrings
        self.output_doc['q'] = "1d ndarray; independent variable"
        self.output_doc['I'] = "1d ndarray; dependent variable; same shape as *q*"
        self.output_doc['dI'] = "1d ndarray; error estimate of *I*; same shape as *I*"
        # source & type
        self.input_src['csv_location'] = optools.fs_input
        self.input_type['csv_location'] = optools.str_type
        self.categories = ['INPUT.CSV']

    def run(self):
        csv_location = self.inputs['csv_location']
        q, I, dI = read_csv_q_I_maybe_dI(csv_location)
        self.outputs['q'] = q
        self.outputs['I'] = I
        self.outputs['dI'] = dI


class FindUnprocessed(Operation):
    """Write q, I, and dI to a csv-formatted file."""

    def __init__(self):
        input_names = ['directory']
        output_names = ['unprocessed']
        super(FindUnprocessed, self).__init__(input_names, output_names)
        # docstrings
        self.input_doc['directory'] = "path to directory whose contents should be checked"
        self.output_doc['unprocessed'] = 'list of unprocessed tif files'
        # source & type
        self.input_src['directory'] = optools.fs_input
        self.input_type['directory'] = optools.str_type
        self.categories = ['MISC']

    def run(self):
        self.outputs['unprocessed'] = find_unprocessed(self.inputs['directory'])


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


def find_by_extension(directory, extension):
    '''
    Find all files in *directory* ending in *extension*.

    :param directory: string path to directory
    :param extension: string extension, e.g. ".txt"
    :return:

    Accepts extensions with or without an initial ".".
    Does not know that tif and tiff are the same thing.
    '''
    # os.splitext gives the extension with '.' in front.
    # Rather than require the user to know this, I take care of it here.
    if extension[0] != '.':
        extension = '.' + extension
    innames = listdir(directory)
    extnames = []
    for ii in range(len(innames)):
        innameii = splitext(innames[ii])
        if innameii[1] == extension:
            extnames.append(join(directory, innames[ii]))
    return extnames


def replace_extension(old_name, new_extension):
    '''
    Return a file name that is identical except for extension.

    :param old_name: string path or file name
    :param new_extension: string extension, e.g. ".txt"
    :return:

    Accepts extensions with or without an initial ".".
    '''
    # os.splitext gives the extension with '.' in front.
    # Rather than require the user to know this, I take care of it here.
    if new_extension[0] != '.':
        new_extension = '.' + new_extension
    root = splitext(old_name)[0]
    new_name = root + new_extension
    return new_name


def read_csv_q_I_maybe_dI(nameloc):
    q = np.loadtxt(nameloc, dtype=float, delimiter=',', skiprows=1, usecols=(0,))
    I = np.loadtxt(nameloc, dtype=float, delimiter=',', skiprows=1, usecols=(1,))
    try:
        dI = np.loadtxt(nameloc, dtype=float, delimiter=',', skiprows=1, usecols=(2,))
    except IndexError:
        dI = np.zeros(1, dtype=float)
    return q, I, dI


def find_unprocessed(directory):
    '''Checks files in a folder to make sure they have been reduced.'''
    images = find_by_extension(directory, 'tif')
    missing_csv = []
    for ii in images:
        csvname = replace_extension(ii, 'csv')
        if not exists(csvname):
            missing_csv.append(ii)
    return missing_csv

#TODO: remove print statement from first_dip
