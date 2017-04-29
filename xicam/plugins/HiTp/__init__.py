from .. import base
import widgets


import numpy as np
from xicam.widgets import daemonwidget


def runtest():
    #EZTest.bottomwidget.clear()
    pass

def openfiles(filepaths):
    '''

    Parameters
    ----------
    filepaths : list
        list of filepaths

    '''

    # TODO: should this line be moved?
    HiTpPlugin.centerwidget.sigPlot.connect(HiTpPlugin.bottomwidget.plot)

    for filepath in filepaths:
        print filepath
        csvpath='?'
        HiTpPlugin.centerwidget.redrawfromCSV(csvpath)
        # handle new file

    #Example of how to get parameter values
    #print HiTpPlugin.parameters.param('ddetect_dist_pix').value()

HiTpPlugin=base.EZplugin(name='HiTp',
                     toolbuttons=[],#('xicam/gui/icons_34.png',runtest)
                     parameters=[{'name':'ddetect_dist_pix','value':2500,'type':'float'},
                                 {'name':'bcenter_x_pix','value':1024,'type':'float'},
                                 {'name':'bcenter_y_pix','value':2500,'type':'float'},
                                 {'name': 'detect_tilt_alpha_rad', 'value': 4.7, 'type': 'float'},
                                 {'name': 'detect_tilt_delta_rad', 'value': 0.5, 'type': 'float'},
                                 {'name': 'wavelength_A', 'value': 0.9762, 'type': 'float'},
                                 {'name': 'polarization', 'value': 0.95, 'type': 'float'},
                                 {'name': 'Imax_Iave_ratio_module', 'value': True, 'type': 'bool'},
                                 {'name': 'texture_module', 'value': True, 'type': 'bool'},
                                 {'name': 'signal_to_noise_module', 'value': True, 'type': 'bool'},
                                 {'name': 'add_feature_to_csv_module', 'value': True, 'type': 'bool'},
                                 daemonwidget.DaemonParameter(openfiles)],
                     openfileshandler=openfiles,
                     centerwidget=widgets.WaferView,
                     bottomwidget=widgets.LocalView)