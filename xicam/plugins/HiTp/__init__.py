"""
Created on Apr 2017

@author: Ron Pandolfi, Fang Ren
"""

from PySide.QtGui import *
from PySide.QtCore import *
from .. import base
from xicam.plugins.HiTp import widgets
from xicam import threads
from xicam import plugins
from create_AIobject import create_AIobject
from xicam.widgets import daemonwidget
from xicam.plugins.HiTp.on_the_fly import run
import os.path
from modpkgs import pyqtgraphmod

def runtest():
    #EZTest.bottomwidget.clear()
    pass

def redrawfromCSV(csvpath):
    print(csvpath)
    plugins.plugins['HiTp'].instance.centerwidget.redrawfromCSV(csvpath)

@threads.method(redrawfromCSV)

def openfiles(filepaths):
    """
    Parameters
    ----------
    filepaths : list
        list of filepaths
    """
    #print(filepaths)

    # calibration
    detect_dist_pix = HiTpPlugin.parameters.param('detect_dist_pix').value()
    bcenter_x_pix = HiTpPlugin.parameters.param('bcenter_x_pix').value()
    bcenter_y_pix = HiTpPlugin.parameters.param('bcenter_y_pix').value()
    detect_tilt_alpha_rad = HiTpPlugin.parameters.param('detect_tilt_alpha_rad').value()
    detect_tilt_beta_rad = HiTpPlugin.parameters.param('detect_tilt_delta_rad').value()
    wavelength_A = HiTpPlugin.parameters.param('wavelength_A').value()
    # first_scan = HiTpPlugin.parameters.param('first_scan').value()
    # last_scan = HiTpPlugin.parameters.param('last_scan').value()
    polarization = HiTpPlugin.parameters.param('polarization').value()

    # parameter
    # first_scan = HiTpPlugin.parameters.param('first_scan').value()
    # last_scan = HiTpPlugin.parameters.param('last_scan').value()
    smpls_per_row = HiTpPlugin.parameters.param('smpls_per_row').value()

    # modules
    Imax_Iave_ratio_module = HiTpPlugin.parameters.param('Imax_Iave_ratio_module').value()
    texture_module = HiTpPlugin.parameters.param('texture_module').value()
    signal_to_noise_module = HiTpPlugin.parameters.param('signal_to_noise_module').value()
    neighbor_distance_module = HiTpPlugin.parameters.param('neighbor_distance_module').value()
    add_feature_to_csv_module = HiTpPlugin.parameters.param('add_feature_to_csv_module').value()
    background_subtract_module = HiTpPlugin.parameters.param('background_subtraction_module').value()
    peak_fitting_module = HiTpPlugin.parameters.param('peak_fitting_module').value()

    # create an AI object for all the processing
    p = create_AIobject(detect_dist_pix, detect_tilt_alpha_rad, detect_tilt_beta_rad, wavelength_A, bcenter_x_pix, bcenter_y_pix)

    for filepath in sorted(filepaths):
        run(filepath, p,
            polarization, smpls_per_row,
            Imax_Iave_ratio_module,
            texture_module,
            signal_to_noise_module,
            neighbor_distance_module,
            add_feature_to_csv_module,
            background_subtract_module,
            peak_fitting_module)

        folder_path, imageFilename = os.path.split(os.path.abspath(filepath))
        csvpath = os.path.join(folder_path, 'Processed//attributes.csv')
    return csvpath


HiTpPlugin=base.EZplugin(name='HiTp',
                     toolbuttons=[],#('xicam/gui/icons_34.png',runtest)
                     parameters=[{'name':'detect_dist_pix','value':2500,'type':'float'}, # calibration tab
                                 {'name':'bcenter_x_pix','value':1024,'type':'float'},
                                 {'name':'bcenter_y_pix','value':2500,'type':'float'},
                                 {'name': 'detect_tilt_alpha_rad', 'value': 4.7, 'type': 'float'},
                                 {'name': 'detect_tilt_delta_rad', 'value': 0.5, 'type': 'float'},
                                 {'name': 'wavelength_A', 'value': 0.9762, 'type': 'float'},
                                 {'name': 'polarization', 'value': 0.95, 'type': 'float'},

                                 # {'name': 'first_scan', 'value': 1, 'type': 'int'}, # parameter tab
                                 # {'name': 'last_scan', 'value': 441, 'type': 'int'},

                                 {'name': 'Imax_Iave_ratio_module', 'value': True, 'type': 'bool'}, # module tab
                                 {'name': 'texture_module', 'value': True, 'type': 'bool'},
                                 {'name': 'signal_to_noise_module', 'value': True, 'type': 'bool'},
                                 {'name': 'add_feature_to_csv_module', 'value': True, 'type': 'bool'},
                                 {'name': 'neighbor_distance_module', 'value': False, 'type': 'bool'},
                                 {'name': 'smpls_per_row', 'value': 25, 'type': 'int'},
                                 {'name': 'background_subtraction_module', 'value': False, 'type': 'bool'},
                                 {'name': 'peak_fitting_module', 'value': False, 'type': 'bool'},
                                 daemonwidget.DaemonParameter(openfiles,filter='*.tif')],
                     openfileshandler=openfiles,
                     centerwidget=widgets.WaferView,
                     bottomwidget=widgets.LocalView)

modes=csvkeys = {'crystallinity':'Imax/Iave','peaks':'num_of_peaks', 'texture':'texture_sum', 'SNR':'SNR', 'NND':'neighbor_distances', 'Imax':'Imax'}

mode = QComboBox(HiTpPlugin.toolbar)
mode.addItems(modes.keys())
HiTpPlugin.toolbar.addWidget(mode)
mode.currentIndexChanged.connect(HiTpPlugin.centerwidget.setMode)