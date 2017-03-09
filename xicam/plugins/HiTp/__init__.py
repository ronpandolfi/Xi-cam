from .. import base



import numpy as np



def runtest():
    EZTest.bottomwidget.clear()

def openfiles(filepaths):
    # handle new file
    pass

EZTest=base.EZplugin(name='HiTp',
                     toolbuttons=[],#('xicam/gui/icons_34.png',runtest)
                     parameters=[{'name':'d_in_pixel','value':2500,'type':'float'},
                                 {'name':'beam_center_x','value':1024,'type':'float'},
                                 {'name':'beam_center_y','value':2500,'type':'float'},
                                 {'name': 'rotation_rad', 'value': 4.7, 'type': 'float'},
                                 {'name': 'tilt_rad', 'value': 0.5, 'type': 'float'},
                                 {'name': 'wavelength_A', 'value': 0.9762, 'type': 'float'},
                                 {'name': 'polarization', 'value': 0.95, 'type': 'float'},
                                 {'name': 'Imax_Iave_ratio_module', 'value': True, 'type': 'boolean'},
                                 {'name': 'texture_module', 'value': True, 'type': 'boolean'},
                                 {'name': 'signal_to_noise_module', 'value': True, 'type': 'boolean'},
                                 {'name': 'add_feature_to_csv_module', 'value': True, 'type': 'boolean'}],
                     openfileshandler=openfiles,
                     centerwidget=None,
                     bottomwidget=None)

