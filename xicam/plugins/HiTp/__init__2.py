from .. import base
import widgets


import numpy as np
from on_the_fly import on_the_fly


def runtest():
    EZTest.bottomwidget.clear()

def openfiles(filepaths):
    # handle new file



    csvpath = '?'
    HiTpPlugin.centerwidget.redrawfromCSV(csvpath)

    # TODO

    detect_dist_pix = HiTpPlugin.parameters.param('detect_dist_pix').value()
    bcenter_x_pix = HiTpPlugin.parameters.param('bcenter_x_pix').value()
    bcenter_y_pix = HiTpPlugin.parameters.param('bcenter_y_pix').value()
    detect_tilt_alpha_rad = HiTpPlugin.parameters.param('detect_tilt_alpha_rad').value()
    detect_tilt_beta_rad = HiTpPlugin.parameters.param('detect_tilt_beta_rad').value()
    wavelength_A = HiTpPlugin.parameters.param('wavelength_A').value()
    first_scan = HiTpPlugin.parameters.param('first_scan').value()
    last_scan = HiTpPlugin.parameters.param('last_scan').value()
    polarization = HiTpPlugin.parameters.param('polarization').value()

    Imax_Iave_ratio_module = HiTpPlugin.parameters.param('Imax_Iave_ratio_module').value()
    texture_module = HiTpPlugin.parameters.param('texture_module').value()
    signal_to_noise_module = HiTpPlugin.parameters.param('signal_to_noise_module').value()
    add_feature_to_csv_module = HiTpPlugin.parameters.param('add_feature_to_csv_module').value()

    on_the_fly(filepaths, first_scan, last_scan,
               detect_dist_pix, detect_tilt_alpha_rad, detect_tilt_beta_rad, wavelength_A, bcenter_x_pix, bcenter_y_pix,
               polarization, num_of_smpls_per_row,
               Imax_Iave_ratio_module,
               texture_module,
               signal_to_noise_module,
               extract_neighbor_distance_module, add_feature_to_csv_module)


HiTpPlugin=base.EZplugin(name='HiTp',
                     toolbuttons=[],#('xicam/gui/icons_34.png',runtest)
                     parameters=[{'name':'detect_dist_pix','value':2500,'type':'float'},
                                 {'name':'bcenter_x_pix','value':1024,'type':'float'},
                                 {'name':'bcenter_y_pix','value':2500,'type':'float'},
                                 {'name': 'detect_tilt_alpha_rad', 'value': 4.7, 'type': 'float'},
                                 {'name': 'detect_tilt_delta_rad', 'value': 0.5, 'type': 'float'},
                                 {'name': 'wavelength_A', 'value': 0.9762, 'type': 'float'},
                                 {'name': 'polarization', 'value': 0.95, 'type': 'float'},
                                 {'name': 'first_scan', 'value': 1, 'type': 'int'},
                                 {'name': 'last_scan', 'value': 441, 'type': 'int'},
                                 {'name': 'polarization', 'value': 0.95, 'type': 'float'},
                                 {'name': 'Imax_Iave_ratio_module', 'value': True, 'type': 'bool'},
                                 {'name': 'texture_module', 'value': True, 'type': 'bool'},
                                 {'name': 'signal_to_noise_module', 'value': True, 'type': 'bool'},
                                 {'name': 'add_feature_to_csv_module', 'value': True, 'type': 'bool'}],
                     openfileshandler=openfiles,
                     centerwidget=widgets.WaferView,
                     bottomwidget=widgets.LocalView)