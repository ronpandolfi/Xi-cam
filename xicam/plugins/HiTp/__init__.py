from .. import base
import widgets


import numpy as np



def runtest():
    EZTest.bottomwidget.clear()

def openfiles(filepaths):
    # handle new file
    pass

EZTest=base.EZplugin(name='HiTp',
                     toolbuttons=[],#('xicam/gui/icons_34.png',runtest)
                     parameters=[{'name':'ddetect_dist_pix','value':2500,'type':'float'},
                                 {'name':'bcenter_x_pix','value':1024,'type':'float'},
                                 {'name':'bcenter_x_pix','value':2500,'type':'float'},
                                 {'name': 'detect_tilt_alpha_rad', 'value': 4.7, 'type': 'float'},
                                 {'name': 'detect_tilt_delta_rad', 'value': 0.5, 'type': 'float'},
                                 {'name': 'wavelength_A', 'value': 0.9762, 'type': 'float'},
                                 {'name': 'polarization', 'value': 0.95, 'type': 'float'},
                                 {'name': 'Imax_Iave_ratio_module', 'value': True, 'type': 'boolean'},
                                 {'name': 'texture_module', 'value': True, 'type': 'boolean'},
                                 {'name': 'signal_to_noise_module', 'value': True, 'type': 'boolean'},
                                 {'name': 'add_feature_to_csv_module', 'value': True, 'type': 'boolean'}],
                     openfileshandler=openfiles,
                     centerwidget=widgets.WaferView,
                     bottomwidget=widgets.LocalView)