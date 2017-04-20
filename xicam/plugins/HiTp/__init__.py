from .. import base
import widgets
from on_the_fly import run
import os.path

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
    for filepath in filepaths:
        # # TODO: should this line be moved?
        # HiTpPlugin.centerwidget.sigPlot.connect(HiTpPlugin.bottomwidget.plot)



        csvpath = filepath[:-8] + 'master.csv'
        HiTpPlugin.centerwidget.redrawfromCSV(csvpath)
        # handle new file
        #Example of how to get parameter values
        #print HiTpPlugin.parameters.param('ddetect_dist_pix').value()


        # calibration
        detect_dist_pix = HiTpPlugin.parameters.param('detect_dist_pix').value()
        bcenter_x_pix = HiTpPlugin.parameters.param('bcenter_x_pix').value()
        bcenter_y_pix = HiTpPlugin.parameters.param('bcenter_y_pix').value()
        detect_tilt_alpha_rad = HiTpPlugin.parameters.param('detect_tilt_alpha_rad').value()
        detect_tilt_beta_rad = HiTpPlugin.parameters.param('detect_tilt_delta_rad').value()
        wavelength_A = HiTpPlugin.parameters.param('wavelength_A').value()
        #first_scan = HiTpPlugin.parameters.param('first_scan').value()
        #last_scan = HiTpPlugin.parameters.param('last_scan').value()
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

        run(filepath, csvpath, detect_dist_pix, detect_tilt_alpha_rad, detect_tilt_beta_rad, wavelength_A,
            bcenter_x_pix, bcenter_y_pix,
            polarization, smpls_per_row,
            Imax_Iave_ratio_module,
            texture_module,
            signal_to_noise_module,
            neighbor_distance_module,
            add_feature_to_csv_module)

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
                                 {'name': 'smpls_per_row', 'value': 25, 'type': 'int'},

                                 {'name': 'Imax_Iave_ratio_module', 'value': True, 'type': 'bool'}, # module tab
                                 {'name': 'texture_module', 'value': True, 'type': 'bool'},
                                 {'name': 'signal_to_noise_module', 'value': True, 'type': 'bool'},
                                 {'name': 'neighbor_distance_module', 'value': False, 'type': 'bool'},
                                 {'name': 'add_feature_to_csv_module', 'value': True, 'type': 'bool'}],
                     openfileshandler=openfiles,
                     centerwidget=widgets.WaferView,
                     bottomwidget=widgets.LocalView)


