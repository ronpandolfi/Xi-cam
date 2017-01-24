from collections import deque
import numpy as np
import tomopy
import pyqtgraph as pg
from PySide import QtGui, QtCore
from collections import OrderedDict
from loader import ProjectionStack, SinogramStack
from pipeline.loader import StackImage
from pipeline import msg
from xicam.plugins.tomography import functionwidgets, reconpkg, config
from xicam.widgets.customwidgets import DataTreeWidget, ImageView, dataDialog
from xicam.widgets.roiwidgets import ROImageOverlay
from xicam.widgets.imageviewers import StackViewer
from xicam.widgets.volumeviewers import VolumeViewer


__author__ = "Luis Barroso-Luque"
__copyright__ = "Copyright 2016, CAMERA, LBL, ALS"
__credits__ = ["Ronald J Pandolfi", "Dinesh Kumar", "Singanallur Venkatakrishnan", "Luis Luque", "Alexander Hexemer"]
__license__ = ""
__version__ = "1.2.1"
__maintainer__ = "Ronald J Pandolfi"
__email__ = "ronpandolfi@lbl.gov"
__status__ = "Beta"


class TomoViewer(QtGui.QWidget):
    """
    Class that holds projection, sinogram, recon preview, and process-settings viewers for a tomography dataset.

    Attributes
    ----------
    data : pipeline.loader.StackImage
        Raw tomography data as a StackImage
    viewstack : QtGui.StackedWidget
        Container for different tomography viewers
    viewmode : QtGui.QTabBar
        Tabbar to switch between tomopgraphy views
    projectionViewer : ProejctionViewer
        Viewer class to visualize raw tomography projections
    sinogramViewer : StackViewer
        Viewer class to visualize raw tomography sinograms
    previewViewer : PreviewViewer
        Viewer class to hold a set of preview reconstructions of a single sinogram/slice
    preview3DViewer : Preview3DViewer
        Viewer class to visualize a reconstruction of subsampled set of the raw data
    pipeline : OrderedDict
        Dictionary to hold parameters for reconstruction, referenced by the iterations
        of the reconstruction function

    Signals
    -------
    sigSetDefaults(dict)
        emits dictionary with dataset specific defaults to set in tomography plugin

    Parameters
    ----------
    paths : str/list of str, optional
        Path to input dataset
    data : ndarry, optional
        Array with input data. Currently only paths are supported.
    args
        Additional arguments
    kwargs
        Additional keyword arguments
    """

    sigSetDefaults = QtCore.Signal(dict)

    def __init__(self, paths=None, data=None, *args, **kwargs):
        if paths is None and data is None:
            raise ValueError('Either data or path to file must be provided')

        super(TomoViewer, self).__init__(*args, **kwargs)

        # pipeline dictionary of parameters
        self.pipeline = OrderedDict()

        # set path as field of TomoViewer
        self.path = paths

        # self._recon_path = None
        self.viewstack = QtGui.QStackedWidget(self)
        self.viewmode = QtGui.QTabBar(self)
        self.viewmode.addTab('Projection View')  # TODO: Add icons!
        self.viewmode.addTab('Sinogram View')
        self.viewmode.addTab('Slice Preview')
        self.viewmode.addTab('3D Preview')
        self.viewmode.setShape(QtGui.QTabBar.TriangularSouth)


        # keep a timer for reconstruction
        self.recon_start_time = 0
        self.preview_holder = []
        self.prange = []




        if data is not None:
            self.data = data
        elif paths is not None and len(paths):
            self.data = self.loaddata(paths)

        if self.data.flats is None and self.data.darks is None:
            import fabio
            flat_dialog = QtGui.QFileDialog(self).getOpenFileName(caption="Flats not detected in input data. Please select flats for this dataset: ")
            dark_dialog = QtGui.QFileDialog(self).getOpenFileName(caption="Darks not detected in input data. Please select darks for this dataset: ")

            if flat_dialog[0] and dark_dialog[0]:
                try:
                    flats = fabio.open(flat_dialog[0])
                    darks = fabio.open(dark_dialog[0])
                    self.data.flats = np.stack([np.copy(flats._dgroup[frame]) for frame in flats.frames])
                    self.data.darks = np.stack([np.copy(darks._dgroup[frame]) for frame in darks.frames])

                    del flats, darks
                except IOError:
                    QtGui.QMessageBox.warning(self, 'Warning','Flats and/or darks not loaded. Cannot perform \
                                                              reconstructions on this data set')
            else:
                QtGui.QMessageBox.warning(self, 'Warning', 'Flats and/or darks not provided. Cannot perform \
                                                          reconstructions on this data set')


        self.projectionViewer = ProjectionViewer(self.data, parent=self)
        self.projectionViewer.centerBox.setRange(0, self.data.shape[1])
        self.viewstack.addWidget(self.projectionViewer)

        self.sinogramViewer = StackViewer(SinogramStack.cast(self.data), parent=self)
        self.sinogramViewer.setIndex(self.sinogramViewer.data.shape[0] // 2)
        self.viewstack.addWidget(self.sinogramViewer)

        self.previewViewer = PreviewViewer(self.data.shape[1], parent=self)
        self.previewViewer.sigSetDefaults.connect(self.sigSetDefaults.emit)
        self.viewstack.addWidget(self.previewViewer)

        self.preview3DViewer = Preview3DViewer(parent=self)
        self.preview3DViewer.volumeviewer.moveGradientTick(1, 0.3)
        self.preview3DViewer.sigSetDefaults.connect(self.sigSetDefaults.emit)
        self.viewstack.addWidget(self.preview3DViewer)


        v = QtGui.QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(self.viewstack)
        v.addWidget(self.viewmode)
        self.setLayout(v)


        self.viewmode.currentChanged.connect(self.viewstack.setCurrentIndex)
        self.viewstack.currentChanged.connect(self.viewmode.setCurrentIndex)

    def wireupCenterSelection(self, recon_function):
        """
        Connect the reconstruction functions parameters to the manual center selection button.
        And connect the parameters sigValueChanged to the center detection image overlay widget

        Parameters
        ----------
        recon_function : FuncionWidget
            Reconstruction function widget with a 'center' child parameter

        """
        if recon_function is not None:
            center_param = recon_function.params.child('center')
            # Uncomment this if you want convenience of having the center parameter in pipeline connected to the
            # manual center widget, but this limits the center options to a resolution of 0.5
            # self.projectionViewer.sigCenterChanged.connect(
            #     lambda x: center_param.setValue(x)) #, blockSignal=center_param.sigValueChanged))
            self.projectionViewer.setCenterButton.clicked.connect(
                lambda: center_param.setValue(self.projectionViewer.centerBox.value()))
            center_param.sigValueChanged.connect(lambda p,v: self.projectionViewer.centerBox.setValue(v))
            center_param.sigValueChanged.connect(lambda p,v: self.projectionViewer.updateROIFromCenter(v))

    @staticmethod
    def loaddata(paths, raw=True):
        """
        Load data from a file or list of files

        Parameters
        ----------
        paths : str/list of str
            Path to files
        raw : bool
            Boolean specifiying it the file is a raw dataset with flats and darks
            (not using this now but can be used for files where flats/darks are in seperate files)

        Returns
        -------
        ProjectionStack, StackImage:
            Class with raw data from file

        """

        if raw:
            return ProjectionStack(paths)
        else:
            return StackImage(paths)


    def getsino(self, slc=None): #might need to redo the flipping and turning to get this in the right orientation
        """
        Returns the sinograms specified in slc (this and getproj can be made one function)

        Parameters
        ----------
        slc : slice, optional
            Slice object specifying the portion of the array to return

        Returns
        -------
        ndarray:
            Array of raw data

        """
        if slc is None:
            return np.ascontiguousarray(self.sinogramViewer.currentdata[:,np.newaxis,:])
        else:
            return np.ascontiguousarray(self.data.fabimage[slc])

    def getproj(self, slc=None):
        """
        Returns the projections specified in slc (this and getsino can be made one function)

        Parameters
        ----------
        slc : slice, optional
            Slice object specifying the portion of the array to return

        Returns
        -------
        ndarray:
            Array of raw data

        """
        if slc is None:
            return np.ascontiguousarray(self.projectionViewer.currentdata[np.newaxis, :, :])
        else:
            return np.ascontiguousarray(self.data.fabimage[slc])

    def getflats(self, slc=None):
        """
        Returns the flat fields specified in slc

        Parameters
        ----------
        slc : slice, optional
            Slice object specifying the portion of the array to return

        Returns
        -------
        ndarray:
            Array of flat field data

        """
        if slc is None:
            return np.ascontiguousarray(self.data.flats[:, self.sinogramViewer.currentIndex, :])
        else:
            return np.ascontiguousarray(self.data.flats[slc])

    def getdarks(self, slc=None):
        """
        Returns the dark fields specified in slc

        Parameters
        ----------
        slc : slice, optional
            Slice object specifying the portion of the array to return

        Returns
        -------
        ndarray:
            Array of dark field data

        """
        if slc is None:
            return np.ascontiguousarray(self.data.darks[: ,self.sinogramViewer.currentIndex, :])
        else:
            return np.ascontiguousarray(self.data.darks[slc])

    def getheader(self):
        """Return the data's header (metadata)"""
        return self.data.header

    def addSlicePreview(self, params, recon, slice_no=None, prange=None):
        """
        Adds a slice reconstruction preview with the corresponding workflow pipeline dictionary to the previewViewer

        Parameters
        ----------
        params : dict
            Pipeline dictionary
        recon : ndarry
            Reconstructed slice
        slice_no :
            Sinogram/slice number reconstructed

        """
        if slice_no is None:
            slice_num = self.sinogramViewer.view_spinBox.value()
            self.previewViewer.addPreview(np.rot90(recon[0],1), params, slice_num)
        elif type(slice_no) is list:
            for item in range(slice_no[1]- slice_no[0]+1):
                self.previewViewer.addPreview(np.rot90(recon[item], 1), params, item+slice_no[0])
        # this block ensures that the previews are added in order if testparamrange is triggered
        elif prange:
            dummy_prange = dict(prange)
            func = dummy_prange.pop('function')
            param = dummy_prange.keys()[0]

            if len(self.prange) < 1 and recon is not None:
                self.prange = prange[param]

            # this if loop and try statement ensure no errors due to the recursion below
            if recon is not None:
                self.preview_holder.append([recon, params, slice_no])
            try:
                top_val = self.prange[0]
            except IndexError:
                pass

            # run through each recon in the preview_holder, and add them to the preview viewer if the top param in
            # prange matches the preview metadata
            for index, rec in enumerate(self.preview_holder):
                for key in rec[1].iterkeys():
                    if func in key:
                        subfunc = rec[1][key].keys()[0]
                        param_val = rec[1][key][subfunc][param]
                if top_val == param_val:
                    self.previewViewer.addPreview(np.rot90(rec[0][0], 1), rec[1], rec[2])
                    self.preview_holder.pop(index)
                    self.prange = np.delete(self.prange, 0)
                    self.addSlicePreview(params, None, slice_no, prange=prange)

        else:
            self.previewViewer.addPreview(np.rot90(recon[0],1), params, slice_no)
        self.viewstack.setCurrentWidget(self.previewViewer)
        msg.clearMessage()

    def add3DPreview(self, params, recon):
        """
        Adds a slice reconstruction preview with the corresponding workflow pipeline dictionary to the preview3DViewer

        Parameters
        ----------
        params : dict
            Pipeline dictionary
        recon : ndarry
            Reconstructed array

        """

        recon = np.flipud(recon)
        self.viewstack.setCurrentWidget(self.preview3DViewer)
        self.preview3DViewer.setPreview(recon, params)
        hist = self.preview3DViewer.volumeviewer.getHistogram()
        max = hist[0][np.argmax(hist[1])]
        self.preview3DViewer.volumeviewer.setLevels([max, hist[0][-1]])

    def onManualCenter(self, active):
        """
        Activates the manual center portion of the ProjectionViewer.
        This is connected to the corresponding toolbar signal

        Parameters
        ----------
        active : bool
            Boolean specifying to activate or not. True activate, False deactivate

        """

        if active:
            self.viewstack.setCurrentWidget(self.projectionViewer)
            self.projectionViewer.showCenterDetection()
            self.projectionViewer.hideMBIR()
        else:
            self.projectionViewer.hideCenterDetection()

    def onMBIR(self, active):


        if active:
            self.viewstack.setCurrentWidget(self.projectionViewer)
            self.projectionViewer.showMBIR()
            self.projectionViewer.hideCenterDetection()
        else:
            self.projectionViewer.hideMBIR()

    def onROIselection(self):
        """
        Shows a rectangular roi to select portion of data to reconstruct. (Not implemented yet)

        Parameters
        ----------
        active : bool
            Boolean specifying to activate or not. True activate, False deactivate

        """
        self.viewstack.setCurrentWidget(self.projectionViewer)
        self.projectionViewer.addROIselection()

class MBIRViewer(QtGui.QWidget):


    def __init__(self, data, path, *args, **kwargs):
        super(MBIRViewer, self).__init__(*args, **kwargs)
        self.mdata = data.header
        if path is list:
            paths = path[0]
        else:
            paths = path
        self.path = paths
        self.data = data
        self.center = 0
        self.cor_detection_funcs = ['Phase Correlation', 'Vo', 'Nelder-Mead']

        self.runButton = QtGui.QPushButton(parent=self)
        self.runButton.setSizePolicy(QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("gui/icons_34.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.runButton.setIcon(icon)
        # self.runButton.setToolTip("Submit MBIR job to NERSC")
        self.runButton.setToolTip("Generate slurm file")

        self.cor_widget = QtGui.QWidget() #parent widget for center of rotation input

        # set up widget for user choice of manual or auto COR detection
        self.cor_Holder = QtGui.QGroupBox('Center of Rotation', parent = self.cor_widget)
        manual_cor = QtGui.QRadioButton('Manually input center of rotation')
        manual_cor.clicked.connect(self.manualCOR)
        auto_cor = QtGui.QRadioButton('Auto-detect center of rotation')
        auto_cor.clicked.connect(self.autoCOR)
        manual_cor.setChecked(True)
        vbox = QtGui.QVBoxLayout()
        vbox.addWidget(manual_cor)
        vbox.addWidget(auto_cor)
        self.cor_Holder.setLayout(vbox)

        # series of widgets for manual COR input
        self.cor_Value = QtGui.QStackedWidget(parent = self.cor_widget)

        self.manual_tab = QtGui.QWidget()
        self.val_box = QtGui.QDoubleSpinBox(parent = self.manual_tab)
        self.val_box.setRange(0,10000)
        self.val_box.setDecimals(1)
        self.val_box.setValue(int(data.shape[1])/2)
        text_label = QtGui.QLabel('Center of Rotation: ', parent = self.manual_tab)
        text_layout = QtGui.QHBoxLayout()
        text_layout.addWidget(text_label)
        text_layout.addWidget(self.val_box)
        self.manual_tab.setLayout(text_layout)

        self.auto_tab = QtGui.QWidget()
        self.auto_tab_layout = QtGui.QVBoxLayout()

        self.cor_function = functionwidgets.FunctionWidget(name="Center Detection", subname="Phase Correlation",
                                package=reconpkg.packages[config.names["Phase Correlation"][1]])
        self.cor_params = pg.parametertree.Parameter.create(name=self.cor_function.name,
                                             children=config.parameters[self.cor_function.subfunc_name], type='group')
        self.cor_param_tree = pg.parametertree.ParameterTree()
        self.cor_param_tree.setMinimumHeight(200)
        self.cor_param_tree.setMinimumWidth(200)
        self.cor_param_tree.setParameters(self.cor_params,showTop = False)
        for key, val in self.cor_function.param_dict.iteritems():
            if key in [p.name() for p in self.cor_params.children()]:
                self.cor_params.child(key).setValue(val)
                self.cor_params.child(key).setDefault(val)

        self.cor_method_box = QtGui.QComboBox()
        self.cor_method_box.currentIndexChanged.connect(self.changeCORfunction)
        for item in self.cor_detection_funcs:
            self.cor_method_box.addItem(item)
        cor_method_label = QtGui.QLabel('COR detection function: ')
        cor_method_layout = QtGui.QHBoxLayout()
        cor_method_layout.addWidget(cor_method_label)
        cor_method_layout.addWidget(self.cor_method_box)

        # import inspect
        # for item in self.cor_detection_funcs:
        #     func = functionwidgets.FunctionWidget(name="Center Detection", subname=item,
        #                         package=reconpkg.packages[config.names[item][1]])
        #     print item
        #     print func.param_dict
        #     print func.exposed_param_dict
        #     print inspect.getargspec(func._function)[0]
        #     print "======================="

        # for param in self.params.children():
            # param.sigValueChanged.connect(self.paramChanged)

        self.auto_tab_layout.addLayout(cor_method_layout)
        self.auto_tab_layout.addWidget(self.cor_param_tree)
        self.auto_tab.setLayout(self.auto_tab_layout)

        # set up COR stackwidget
        self.cor_Value.addWidget(self.manual_tab)
        self.cor_Value.addWidget(self.auto_tab)

        # set up COR widget
        v = QtGui.QVBoxLayout()
        v.addWidget(self.cor_Holder)
        v.addWidget(self.cor_Value)
        self.cor_widget.setLayout(v)


        self.runButton.clicked.connect(self.write_slurm)

        self.mbirParams = pg.parametertree.ParameterTree()
        self.mbirParams.setMinimumHeight(230)
        params = [{'name': 'Dataset path', 'type': 'str'},
                  {'name': 'Z start', 'type': 'int', 'value': 0, 'default': 0},
                  {'name': 'Z num elts', 'type': 'int', 'value': int(data.shape[-1]) ,
                   'default': int(data.shape[-1])},
                  {'name': 'Smoothness', 'type': 'float', 'value': 0.15, 'default': 0.15},
                  {'name': 'Zinger thresh', 'type': 'float', 'value': 5, 'default': 5},
                  {'name': 'View subsample factor', 'type': 'int', 'value': 2, 'default': 2},
                  {'name': 'Output folder', 'type':'str', 'value':'Results', 'default': 'Results'}]

        self.mbir_params = pg.parametertree.Parameter.create(name='MBIR Parameters', type='group', children=params)
        self.mbirParams.setParameters(self.mbir_params,showTop=False)


        right_menu = QtGui.QSplitter(self)
        right_menu.setOrientation(QtCore.Qt.Vertical)
        button_holder = QtGui.QStackedWidget()
        button_holder.addWidget(self.runButton)
        right_menu.addWidget(button_holder)
        right_menu.addWidget(self.cor_widget)

        container = QtGui.QWidget()
        container_layout = QtGui.QVBoxLayout()
        container_layout.addWidget(right_menu)
        container.setLayout(container_layout)

        left_menu = QtGui.QSplitter(self)
        left_menu.addWidget(self.mbirParams)
        left_menu.addWidget(container)


        h = QtGui.QHBoxLayout()
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(left_menu)


        self.setLayout(h)

    def changeCORfunction(self, index):

        subname = self.cor_method_box.itemText(index)
        self.auto_tab_layout.removeWidget(self.cor_param_tree)

        self.cor_function = functionwidgets.FunctionWidget(name="Center Detection", subname=subname,
                                package=reconpkg.packages[config.names[subname][1]])
        self.cor_params = pg.parametertree.Parameter.create(name=self.cor_function.name,
                                             children=config.parameters[self.cor_function.subfunc_name], type='group')
        self.cor_param_tree = pg.parametertree.ParameterTree()
        self.cor_param_tree.setMinimumHeight(200)
        self.cor_param_tree.setMinimumWidth(200)
        self.cor_param_tree.setParameters(self.cor_params,showTop = False)
        for key, val in self.cor_function.param_dict.iteritems():
            if key in [p.name() for p in self.cor_params.children()]:
                self.cor_params.child(key).setValue(val)
                self.cor_params.child(key).setDefault(val)

        self.auto_tab_layout.addWidget(self.cor_param_tree)
        self.auto_tab.setLayout(self.auto_tab_layout)



    def manualCOR(self):
        self.cor_Value.setCurrentWidget(self.manual_tab)

    def autoCOR(self):
        self.cor_Value.setCurrentWidget(self.auto_tab)

    def loadCOR(self):


        widget = self.cor_Value.currentWidget()
        if widget is self.manual_tab:
            return self.val_box.value()
        else:
            if self.parentWidget():
                return self.find_COR(self.cor_function.subfunc_name)
            else:
                return -1

    def find_COR(self, cor_function):
        if not cor_function in self.cor_detection_funcs:
            return -1
        else:
            if cor_function == 'Phase Correlation':
                proj1, proj2 = map(self.data.fabimage.__getitem__, (0,-1))
                kwargs = {'proj1' : proj1, 'proj2' : proj2}
            elif cor_function == 'Vo':
                kwargs = {'tomo' : np.ascontiguousarray(self.data.fabimage[:, :, :])}
            elif cor_function == 'Nelder-Mead':
                kwargs = {'tomo' : np.ascontiguousarray(self.data.fabimage[:, :, :]),
                          'theta' : tomopy.angles(int(self.data.shape[0]),ang1=90,ang2=270)}
            else:
                return -1

            for child in self.cor_params.children():
                kwargs[child.name()] = child.value()

            val = self.cor_function._function(**kwargs)
            return val[0] if val is list else val



    def write_slurm(self):
        """
        A 'slurm' file is a job to run on nersc
        """

        import os.path
        msg.showMessage("Generating slurm file...", timeout=0)

        self.center = self.loadCOR()

        if not self.center > 0 or self.center > self.data.shape[0]:
            msg.showMessage('Invalid center of rotation')
            pass
        else:

            views = int(self.data[0]) - 1
            file_name = self.path.split("/")[-1].split(".")[0]
            nodes = int(np.ceil(self.mbir_params.child('Z num elts').value()/ float(24)))
            output = os.path.join('/',self.mbir_params.child('Output folder').value(), file_name + '_mbir')

            try:
                group = self.mdata['archdir'].split("\\")[-1]
                px_size = float(self.mdata['pzdist'])*1000
            except KeyError:
                msg.showMessage('Insufficient metadata to write slurm file.', timeout=0)
            if group != file_name:
                group_hdf5 = "{}/{}".format(group, file_name)
            else:
                group_hdf5 = file_name

            slurm = '#!/bin/tcsh\n#SBATCH -p regular\n#SBATCH -N {}\n'.format(nodes)
            slurm += '#SBATCH -t 4:00:00\n#SBATCH -J {}\n#SBATCH -e {}.err\n#SBATCH -o {}.out\n\n'.format(file_name, file_name, file_name)
            slurm += 'setenv OMP_NUM_THREADS 24\nsetenv CRAY_ROOTFS DSL\nmodule load PrgEnv-intel\n'
            slurm += 'module load python/2.7.3\nmodule load h5py\nmodule load pil\nmodule load mpi4py\n\n'
            slurm += 'mkdir $SCRATCH/LaunchFolder\nmkdir $SCRATCH/Results\n\n'
            slurm += 'python XT_MBIR_3D.py --setup_launch_folder --run_reconstruction --Edison'
            slurm += ' --input_hdf5 {}/{}.h5'.format(self.mbir_params.child('Dataset path').value(), file_name)
            slurm += ' --group_hdf5 /{}'.format(group_hdf5)
            slurm += ' --code_launch_folder $SCRATCH/LaunchFolder/'
            slurm += ' --output_hdf5 $SCRATCH/Results/{}_mbir/ --x_width {}'.format(file_name, int(self.data.shape[0]))
            slurm += ' --recon_x_width {} --num_dark {}'.format(str(int(self.data.shape[0])), str(len(self.data.fabimage.darks)))
            slurm += ' --num_bright {} --z_numElts {}'.format(str(self.data.fabimage.flats),self.mbir_params.child('Z num elts').value())
            slurm += ' --z_start {} --num_views {}'.format(self.mbir_params.child('Z start').value(), views)
            slurm += ' --pix_size {} --rot_center {}'.format(px_size, self.center)
            slurm += ' --smoothness {} --zinger_thresh {}'.format(self.mbir_params.child('Smoothness').value(),
                                                                 self.mbir_params.child('Zinger thresh').value())
            slurm += ' --Variance_Est 1 --num_threads 24 --num_nodes {} '.format(nodes)
            slurm += '--view_subsmpl_fact {}'.format(self.mbir_params.child('View subsample factor').value())



            parent_folder = self.path.split(self.path.split('/')[-1])[0]
            write = os.path.join(parent_folder, '{}.slurm'.format(file_name))

            with open(write, 'w') as job:
                job.write(slurm)

            msg.showMessage("Done.", timeout=0)



class ProjectionViewer(QtGui.QWidget):
    """
    Class that holds a stack viewer, an ROImageOverlay and a few widgets to allow manual center detection

    Attributes
    ----------
    stackViewer : StackViewer
        widgets.StackViewer used to display the data
    data : loader.StackImage
        Image data
    flat : ndarray
        Median of flat field data
    dark : ndarray
        Median of dark field data
    imageoverlay_roi : widgets.ROIImageOverlay
        Widget used in the cor_widget for manual center detection
    selection_roi : pyqtgragh.ROI
        ROI for selecting region to reconstruct (Not implemented)
    cor_widget : QtGui.QWidget
        Widget used in manual center detection
    setCenterButton : QtGui.QToolButton
        Button for setting center value from cor_widget to reconstruction function in pipeline

    Signals
    -------
    sigCenterChanged(float)
        emits float with new center value

    Parameters
    ----------
    data : pipeline.loader.StackImage
        Raw tomography data as a StackImage
    view_label : str
        String to show in QLabel lower right hand corner. Where the current index is displayed
    center : float
        center of rotation value
    args
        Additional arguments
    kwargs
        Additional keyword arguments
    """

    sigCenterChanged = QtCore.Signal(float)

    def __init__(self, data, view_label=None, center=None, paths=None, *args, **kwargs):
        super(ProjectionViewer, self).__init__(*args, **kwargs)



        self.stackViewer = StackViewer(data, view_label=view_label)
        self.imageItem = self.stackViewer.imageItem
        self.data = self.stackViewer.data
        self.normalized = False
        # self.flat = np.median(self.data.flats, axis=0).transpose()
        # self.dark = np.median(self.data.darks, axis=0).transpose()
        self.imgoverlay_roi = ROImageOverlay(self.data, self.imageItem, [0, 0], parent=self.stackViewer.view)
        self.imageItem.sigImageChanged.connect(self.imgoverlay_roi.updateImage)
        self.stackViewer.view.addItem(self.imgoverlay_roi)
        self.roi_histogram = pg.HistogramLUTWidget(image=self.imgoverlay_roi.imageItem, parent=self.stackViewer)
        self.mbir_viewer = MBIRViewer(self.data, path = self.parentWidget().path, parent=self)


        # roi to select region of interest
        self.selection_roi = None

        self.stackViewer.ui.gridLayout.addWidget(self.roi_histogram, 0, 3, 1, 2)
        self.stackViewer.keyPressEvent = self.keyPressEvent

        self.cor_widget = QtGui.QWidget(self)
        clabel = QtGui.QLabel('Rotation Center:')
        olabel = QtGui.QLabel('Offset:')
        self.centerBox = QtGui.QDoubleSpinBox(parent=self.cor_widget) #QtGui.QLabel(parent=self.cor_widget)
        self.centerBox.setDecimals(1)
        self.setCenterButton = QtGui.QToolButton()
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("xicam/gui/icons_45.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.setCenterButton.setIcon(icon)
        self.setCenterButton.setToolTip('Set center in pipeline')
        originBox = QtGui.QLabel(parent=self.cor_widget)
        originBox.setText('x={}   y={}'.format(0, 0))
        center = center if center is not None else data.shape[1]/2.0
        self.centerBox.setValue(center) #setText(str(center))
        h1 = QtGui.QHBoxLayout()
        h1.setAlignment(QtCore.Qt.AlignLeft)
        h1.setContentsMargins(0, 0, 0, 0)
        h1.addWidget(clabel)
        h1.addWidget(self.centerBox)
        h1.addWidget(self.setCenterButton)
        h1.addWidget(olabel)
        h1.addWidget(originBox)

        plabel = QtGui.QLabel('Overlay Projection No:')
        plabel.setAlignment(QtCore.Qt.AlignRight)
        spinBox = QtGui.QSpinBox(parent=self.cor_widget)
        #TODO data shape seems to be on larger than the return from slicing it with [:-1]
        spinBox.setRange(0, data.shape[0])
        slider = QtGui.QSlider(orientation=QtCore.Qt.Horizontal, parent=self.cor_widget)
        slider.setRange(0, data.shape[0])
        spinBox.setValue(data.shape[0])
        slider.setValue(data.shape[0])
        flipCheckBox = QtGui.QCheckBox('Flip Overlay', parent=self.cor_widget)
        flipCheckBox.setChecked(True)
        constrainYCheckBox = QtGui.QCheckBox('Constrain Y', parent=self.cor_widget)
        constrainYCheckBox.setChecked(True)
        constrainXCheckBox = QtGui.QCheckBox('Constrain X', parent=self.cor_widget)
        constrainXCheckBox.setChecked(False)
        # rotateCheckBox = QtGui.QCheckBox('Enable Rotation', parent=self.cor_widget)
        # rotateCheckBox.setChecked(False)
        self.normCheckBox = QtGui.QCheckBox('Normalize', parent=self.cor_widget)
        h2 = QtGui.QHBoxLayout()
        h2.setAlignment(QtCore.Qt.AlignLeft)
        h2.setContentsMargins(0, 0, 0, 0)
        h2.addWidget(plabel)
        h2.addWidget(spinBox)
        h2.addWidget(flipCheckBox)
        h2.addWidget(constrainXCheckBox)
        h2.addWidget(constrainYCheckBox)
        # h2.addWidget(rotateCheckBox) # This needs to be implemented correctly
        h2.addWidget(self.normCheckBox)
        h2.addStretch(1)
        spinBox.setFixedWidth(spinBox.width())
        v = QtGui.QVBoxLayout(self.cor_widget)
        v.addLayout(h1)
        v.addLayout(h2)
        v.addWidget(slider)

        l = QtGui.QGridLayout(self)
        l.setContentsMargins(0, 0, 0, 0)
        l.addWidget(self.cor_widget)
        l.addWidget(self.stackViewer)
        l.addWidget(self.mbir_viewer)
        self.hideMBIR()
        # self.mbir_viewer.hide()

        slider.valueChanged.connect(spinBox.setValue)
        slider.valueChanged.connect(self.stackViewer.resetImage)
        spinBox.valueChanged.connect(self.changeOverlayProj)
        flipCheckBox.stateChanged.connect(self.flipOverlayProj)
        constrainYCheckBox.stateChanged.connect(lambda v: self.imgoverlay_roi.constrainY(v))
        constrainXCheckBox.stateChanged.connect(lambda v: self.imgoverlay_roi.constrainX(v))

        # rotateCheckBox.stateChanged.connect(self.addRotateHandle)
        self.normCheckBox.stateChanged.connect(self.normalize)
        self.stackViewer.sigTimeChanged.connect(lambda: self.normalize(False))
        self.imgoverlay_roi.sigTranslated.connect(self.setCenter)
        self.imgoverlay_roi.sigTranslated.connect(lambda x, y: originBox.setText('x={}   y={}'.format(x, y)))
        self.hideCenterDetection()

        self.bounds = None
        # self.normalize(True)


    def changeOverlayProj(self, idx):
        """
        Changes the image in the overlay. This is connected to the slider in the cor_widget
        """

        self.normCheckBox.setChecked(False)
        self.imgoverlay_roi.setCurrentImage(idx)
        self.imgoverlay_roi.updateImage()

    def setCenter(self, x, y):
        """
        Sets the center in the centerBox based on the position of the imageoverlay

        Parameters
        ----------
        x : float
            x-coordinate of overlay image in the background images coordinates
        y : float
            x-coordinate of overlay image in the background images coordinates
        """

        center = (self.data.shape[1] + x - 1)/2.0 # subtract half a pixel out of 'some' convention?
        self.centerBox.setValue(center)
        self.sigCenterChanged.emit(center)

    def hideCenterDetection(self):
        """
        Hides the center detection widget and corresponding histogram
        """
        self.normalize(False)
        self.cor_widget.hide()
        self.roi_histogram.hide()
        self.imgoverlay_roi.setVisible(False)

    def showCenterDetection(self):
        """
        Shows the center detection widget and corresponding histogram
        """
        # self.normalize(True)
        self.cor_widget.show()
        self.roi_histogram.show()
        self.imgoverlay_roi.setVisible(True)

    def showMBIR(self):
        self.mbir_viewer.show()
        # self.hideCenterDetection()
        self.stackViewer.hide()

    def hideMBIR(self):
        self.mbir_viewer.hide()
        self.stackViewer.show()


    def updateROIFromCenter(self, center):
        """
        Updates the position of the ROIImageOverlay based on the given center

        Parameters
        ----------
        center : float
            Location of center of rotation
        """

        s = self.imgoverlay_roi.pos()[0]
        self.imgoverlay_roi.translate(pg.Point((2 * center + 1 - self.data.shape[1] - s, 0))) # 1 again due to the so-called COR
                                                                                   # conventions...
    def flipOverlayProj(self, val):
        """
        Flips the image show in the ROIImageOverlay
        """

        self.imgoverlay_roi.flipCurrentImage()
        self.imgoverlay_roi.updateImage()

    def toggleRotateHandle(self, val):
        """
        Adds/ removes a handle on the ROIImageOverlay to be able to rotate the image (Rotation is not implemented
        correctly yet)

        Parameters
        ----------
        val : bool
            Boolean specifying to add or remove the handle
        """

        if val:
            self.toggleRotateHandle.handle = self.imgoverlay_roi.addRotateHandle([0, 1], [0.2, 0.2])
        else:
            self.imgoverlay_roi.removeHandle(self.toggleRotateHandle.handle)

    def addROIselection(self):
        """
        Adds/ removes a rectangular ROI to select a region of interest for reconstruction. Not implemented yet
        """

        self.selection_roi = pg.ROI([0, 0], [10, 10])
        self.stackViewer.view.addItem(self.selection_roi)
        self.selection_roi.addScaleHandle([1, 1], [0, 0])
        self.selection_roi.addScaleHandle([0, 0], [1, 1])

    def normalize(self, val):
        """
        Toggles the normalization of the ROIImageOverlay.

        Parameters
        ----------
        val : bool
            Boolean specifying to normalize image
        """
        if val and not self.normalized:
            self.flat = np.median(self.data.flats, axis=0).transpose()
            self.dark = np.median(self.data.darks, axis=0).transpose()

            proj = (self.imageItem.image - self.dark)/(self.flat - self.dark)
            overlay = self.imgoverlay_roi.currentImage
            if self.imgoverlay_roi.flipped:
                overlay = np.flipud(overlay)
            overlay = (overlay - self.dark)/(self.flat - self.dark)
            if self.imgoverlay_roi.flipped:
                overlay = np.flipud(overlay)
            self.imgoverlay_roi.currentImage = overlay

            # TODO: change roi default levels during normalization to prevent washed out color
            # if not self.bounds:
            #     hist = self.imgoverlay_roi.imageItem.getHistogram()
            #     arr1, arr2 = self.imgoverlay_roi.remove_outlier(hist[1], hist[0], sp.integrate.trapz(hist[1],hist[0]),
            #                                                     thresh=0.4)
            #     print len(arr1), ", ", len(arr2)
            #     self.bounds = [arr2[0],arr2[-1]]
            #     print self.bounds

            self.imgoverlay_roi.updateImage(autolevels=True)
            self.stackViewer.setImage(proj, autoRange=False, autoLevels=True)
            self.stackViewer.updateImage()
            self.normalized = True
            self.normCheckBox.setChecked(True)
        elif not val and self.normalized:
            self.stackViewer.resetImage()
            self.imgoverlay_roi.resetImage()
            self.normalized = False
            self.normCheckBox.setChecked(False)

    def keyPressEvent(self, ev):
        """
        Override QWidgets key pressed event to send the event to the ROIImageOverlay when it is pressed
        """
        super(ProjectionViewer, self).keyPressEvent(ev)
        if self.imgoverlay_roi.isVisible():
            self.imgoverlay_roi.keyPressEvent(ev)
        else:
            super(StackViewer, self.stackViewer).keyPressEvent(ev)
        ev.accept()


class PreviewViewer(QtGui.QSplitter):
    """
    Viewer class to show reconstruction previews in a PG ImageView, along with the function pipeline settings for the
    corresponding preview

    Attributes
    ----------
    previews : ArrayDeque
        ArrayDeque to hold slice reconstruction previews
    data : deque of dicts
        deque holding preview dicts corresponding to the reconstruction in previews
    datatrees : deque of DataTree widgets
        deque holding DataTree widgets to show the data in data deque
    slice_numbers : deque
        deque with sinogram index that was reconstructed for that preview
    imageview : widgets.ImageView
        ImageView to display preview reconstructions

    Signals
    -------
    sigSetDefaults(dict)
        Emits dictionary of current preview. Used to set the workflow pipeline according to the emitted dict

    Parameters
    ----------
    dim : int
        Dimensions of arrays in preview array deque. This is no longer used because array deque can hold arrays of
        different size.
    maxpreviews : int
        Maximum number of preview arrrays that can be held
    args
    kwargs
    """

    sigSetDefaults = QtCore.Signal(dict)

    def __init__(self, dim, maxpreviews=None, *args, **kwargs):
        super(PreviewViewer, self).__init__(*args, **kwargs)
        self.maxpreviews = maxpreviews if maxpreviews is not None else 40

        self.dim = dim

        self.previews = ArrayDeque(arrayshape=(dim, dim), maxlen=self.maxpreviews)
        self.datatrees = deque(maxlen=self.maxpreviews)
        self.data = deque(maxlen=self.maxpreviews)
        self.slice_numbers = deque(maxlen=self.maxpreviews)

        self.setOrientation(QtCore.Qt.Horizontal)
        self.functionform = QtGui.QStackedWidget()

        self.deleteButton = QtGui.QToolButton(self)
        self.deleteButton.setToolTip('Delete this preview')
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("xicam/gui/icons_36.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.deleteButton.setIcon(icon)

        self.setPipelineButton = QtGui.QToolButton(self)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("xicam/gui/icons_45.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.setPipelineButton.setIcon(icon)
        self.setPipelineButton.setToolTip('Set as pipeline')

        ly = QtGui.QVBoxLayout()
        ly.setContentsMargins(0, 0, 0, 0)
        ly.setSpacing(0)
        ly.addWidget(self.functionform)
        h = QtGui.QHBoxLayout()
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(self.setPipelineButton)
        h.addWidget(self.deleteButton)
        ly.addLayout(h)
        panel = QtGui.QWidget(self)
        panel.setLayout(ly)
        self.setPipelineButton.hide()
        self.deleteButton.hide()

        self.imageview = ImageView(self)
        self.imageview.ui.roiBtn.setParent(None)
        self.imageview.ui.roiBtn.setParent(None)
        self.imageview.ui.menuBtn.setParent(None)

        self.view_label = QtGui.QLabel(self)
        self.view_label.setText('No: ')
        self.view_number = QtGui.QSpinBox(self)
        self.view_number.setReadOnly(True)
        self.view_number.setMaximum(5000) # Large enough number
        self.imageview.ui.gridLayout.addWidget(self.view_label, 1, 1, 1, 1)
        self.imageview.ui.gridLayout.addWidget(self.view_number, 1, 2, 1, 1)

        self.setCurrentIndex = self.imageview.setCurrentIndex
        self.addWidget(panel)
        self.addWidget(self.imageview)

        self.imageview.sigDeletePressed.connect(self.removePreview)
        self.setPipelineButton.clicked.connect(self.defaultsButtonClicked)
        self.deleteButton.clicked.connect(self.removePreview)
        self.imageview.sigTimeChanged.connect(self.indexChanged)

    @ QtCore.Slot(object, object)
    def indexChanged(self, index, time):
        """Slot connected to the ImageViews sigChanged"""
        try:
            self.functionform.setCurrentWidget(self.datatrees[index])
            self.view_number.setValue(self.slice_numbers[index])
        except IndexError as e:
            print 'index {} does not exist'.format(index)

    # Could be leaking memory if I don't explicitly delete the datatrees that are being removed
    # from the previewdata deque but are still in the functionform widget? Hopefully python gc is taking good care of me
    def addPreview(self, image, funcdata, slice_number):
        """
        Adds a preview

        Parameters
        ----------
        image : ndarray
            Reconstructed image
        funcdata : dict
            Dictionary summarizing pipeline used for reconstruction
        slice_number : int
            Index of sinogram reconstructed
        """

        self.deleteButton.show()
        self.setPipelineButton.show()
        self.previews.appendleft(np.flipud(image))
        functree = DataTreeWidget()
        functree.setHeaderHidden(True)
        functree.setData(funcdata, hideRoot=True)
        functree.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
        functree.setSelectionBehavior(QtGui.QAbstractItemView.SelectItems)

        self.data.appendleft(funcdata)
        self.datatrees.appendleft(functree)
        self.slice_numbers.appendleft(slice_number)
        self.view_number.setValue(slice_number)
        self.functionform.addWidget(functree)
        levels = False if len(self.data) > 1 else True
        self.imageview.setImage(self.previews, autoRange=False, autoLevels=levels, autoHistogramRange=False)
        self.functionform.setCurrentWidget(functree)

    def removePreview(self):
        """
        Removes the current preview
        """
        if len(self.previews) > 0:
            idx = self.imageview.currentIndex
            self.functionform.removeWidget(self.datatrees[idx])
            del self.previews[idx]
            del self.datatrees[idx]
            del self.data[idx]
            del self.slice_numbers[idx]
            if len(self.previews) == 0:
                self.imageview.clear()
                self.deleteButton.hide()
                self.setPipelineButton.hide()
            else:
                self.imageview.setImage(self.previews)

    def defaultsButtonClicked(self):
        """
        Emits the dict of current preview
        """
        current_data = self.data[self.imageview.currentIndex]
        self.sigSetDefaults.emit(current_data)


class Preview3DViewer(QtGui.QSplitter):
    """
    Viewer class to show 3D reconstruction previews, along with the function pipeline settings for the
    corresponding preview


    Attributes
    ----------
    volumviewer : widgets,volumeviewers.VolumeViewer
        VolumeViewer widget to render 3D preview reconstruction volume
    fdata : dict
        dict corresponding to the reconstruction functions
    pipelinetree : DataTree widget
        Datatree for displaying data dict
    data : ndarray
        Array of reconstructed volume

    Signals
    -------
    sigSetDefaults(dict)
        Emits dictionary of preview. Used to set the workflow pipeline according to the emitted dict

    """

    sigSetDefaults = QtCore.Signal(dict)

    def __init__(self, *args, **kwargs):
        super(Preview3DViewer, self).__init__(*args, **kwargs)
        self.setOrientation(QtCore.Qt.Horizontal)
        l = QtGui.QVBoxLayout()
        l.setContentsMargins(0, 0, 0, 0)
        self.pipelinetree = DataTreeWidget()
        self.pipelinetree.setHeaderHidden(True)
        self.pipelinetree.clear()

        self.setPipelineButton = QtGui.QToolButton(self)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("xicam/gui/icons_45.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.setPipelineButton.setIcon(icon)
        self.setPipelineButton.setToolTip('Set as pipeline')

        ly = QtGui.QVBoxLayout()
        ly.setContentsMargins(0, 0, 0, 0)
        ly.setSpacing(0)
        ly.addWidget(self.pipelinetree)
        h = QtGui.QHBoxLayout()
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(self.setPipelineButton)
        ly.addLayout(h)
        panel = QtGui.QWidget(self)
        panel.setLayout(ly)

        self.volumeviewer = VolumeViewer()

        self.addWidget(panel)
        self.addWidget(self.volumeviewer)

        self.data = None

        self.setPipelineButton.clicked.connect(lambda: self.sigSetDefaults.emit(self.data))
        self.setPipelineButton.hide()

    def setPreview(self, recon, funcdata):
        """
        Sets the 3D preview

        Parameters
        ----------
        recon : ndarray
            3D array of reconstructed volume
        funcdata : dict
            Dictionary summarizing pipeline used for reconstruction
        """

        self.pipelinetree.setData(funcdata, hideRoot=True)
        self.data = funcdata
        self.pipelinetree.show()
        self.volumeviewer.setVolume(vol=recon)
        self.setPipelineButton.show()


class RunConsole(QtGui.QTabWidget):
    """
    Class to output status of a running job, and cancel the job.  Has tab for local run settings
    and can add tabs tab for remote job settings.

    Attributes
    ----------
    local_console : QtGui.QWidget
        Widget for console used when running local reconstructions
    """

    icon = QtGui.QIcon()
    icon.addPixmap(QtGui.QPixmap("xicam/gui/icons_51.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
    icon_clear = QtGui.QIcon()
    icon_clear.addPixmap(QtGui.QPixmap("xicam/gui/icons_57.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)

    def __init__(self, parent=None):
        super(RunConsole, self).__init__(parent=parent)
        self.setTabPosition(QtGui.QTabWidget.West)
        # Text Browser for local run console
        self.local_console, self.local_cancelButton, self.local_clearButton = self.addConsole('Local')
        self.local_clearButton.clicked.connect(self.local_console.clear)

    def addConsole(self, name):
        """
        Adds a new console, This will come in handy when running remote operations (ie adding a console for remote
        location). Will probably need to create an attribute (dict) for holding the names, consoles when adding consoles

        Parameters
        ----------
        name : str
            Name to be used in tab for console added
        """

        # TODO: finish adding message clear button
        console = QtGui.QTextEdit()
        button = QtGui.QToolButton()
        button_clear = QtGui.QToolButton()

        console.setObjectName(name)
        console.setReadOnly(True)
        console.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Preferred)
        button.setIcon(self.icon)
        button.setIconSize(QtCore.QSize(24, 24))
        button.setFixedSize(32, 32)
        button.setToolTip('Cancel running process')
        button_clear.setIcon(self.icon_clear)
        button_clear.setIconSize(QtCore.QSize(24,24))
        button_clear.setFixedSize(32,32)
        button_clear.setToolTip('Clear console log')

        w = QtGui.QWidget()
        w.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Preferred)
        w.setContentsMargins(0, 0, 0, 0)
        l = QtGui.QGridLayout()
        l.setContentsMargins(0, 0, 0, 0)
        l.setSpacing(0)
        l.addWidget(console, 0, 0, 3, 2)
        l.addWidget(button, 0, 2, 1, 1)
        l.addWidget(button_clear,1,2,1,1)
        w.setLayout(l)
        self.addTab(w, console.objectName())
        return console, button, button_clear

    def log2local(self, msg):
        """
        Logs a message to the local console. If adding new consoles there will have to be a way to dynamically create
        a function like this for the added console.
        """
        text = self.local_console.toPlainText()
        if '\n' not in msg:
            self.local_console.setText(msg + '\n\n' + text)
        else:
            topline = text.splitlines()[0]
            tail = '\n'.join(text.splitlines()[1:])
            self.local_console.setText(topline + msg + tail)


class ArrayDeque(deque):
    """
    Class for a numpy array deque where arrays can be appended on both ends.

    Parameters
    ----------
    arraylist : list of ndarrays, optional
        List of ndarrays to initialize
    arrayshape : tuple, optional
        Shape of ndarrays to be held. This is not lenient and not needed since arrays of different sizes can be added
    dtype : type
        Type of array data
    maxlen : int
        Maximum number of arrays that can be held in ArrayDeque
    """

    def __init__(self, arraylist=[], arrayshape=None, dtype=None, maxlen=None):
        # perhaps will need to add check of datatype everytime a new array is added with extend, append, etc??
        if not arraylist and not arrayshape:
            raise ValueError('One of arraylist or arrayshape must be specified')

        super(ArrayDeque, self).__init__(iterable=arraylist, maxlen=maxlen)

        self._shape = [len(self)]
        self._dtype = dtype

        if arraylist:
            # if False in [np.array_equal(arraylist[0].shape, array.shape) for array in arraylist[1:]]:
            #     raise ValueError('All arrays in arraylist must have the same dimensions')
            # elif False in [arraylist[0].dtype == array.dtype for array in arraylist[1:]]:
            #     raise ValueError('All arrays in arraylist must have the same data type')
            map(self._shape.append, arraylist[0].shape)
        elif arrayshape:
            map(self._shape.append, arrayshape)

        self.ndim = len(self._shape)

    @property
    def shape(self):
        """
        Return the shape of the deque based on number of arrays held
        """
        self._shape[0] = len(self)
        return self._shape

    @property
    def size(self):
        """
        Return the size of the array based on number of arrays held
        """
        return np.product(self.shape)

    @property
    def dtype(self):
        """
        Return the dataype of the array's in deque
        """
        if self._dtype is None and self.shape[0]:
            self._dtype = self.__getitem__(0).dtype
        return self._dtype

    @property
    def max(self):
        """
        Return the maximum value
        """
        return np.max(max(self, key=lambda x:np.max(x)))

    @property
    def min(self):
        """
        Return the minimum value
        """
        return np.min(min(self, key=lambda x:np.min(x)))

    def append(self, arr):
        """
        Appends an array to the end of the array deque
        """

        # if arr.shape != tuple(self.shape[1:]):
        #     raise ValueError('Array shape must be {0}, got shape {1}'.format(self.shape[1:], arr.shape))
        # if self.dtype is not None and arr.dtype != self.dtype:
        #     raise ValueError('Array must be of type {}'.format(self.dtype))
        super(ArrayDeque, self).append(arr)

    def appendleft(self, arr):
        """
        Appends an array to the beginning of the array deque
        """

        # if arr.shape != tuple(self.shape[1:]):
        #     raise ValueError('Array shape must be {0}, got shape {1}'.format(self.shape[1:], arr.shape))
        # if self.dtype is not None and arr.dtype != self.dtype:
        #     raise ValueError('Array must be of type {}'.format(self.dtype))
        super(ArrayDeque, self).appendleft(arr)

    def __getitem__(self, item):
        """
        Override slicing
        """
        if type(item) is list and isinstance(item[0], slice):
            dq_item = item.pop(0)
            if isinstance(dq_item, slice):
                dq_item = dq_item.stop if dq_item.stop is not None else dq_item.start if dq_item.start is not None else 0
            return super(ArrayDeque, self).__getitem__(dq_item).__getitem__(item)
        else:
            return super(ArrayDeque, self).__getitem__(item)


# Testing
if __name__ == '__main__':
    import sys

    app = QtGui.QApplication(sys.argv)
    w = RunConsole()
    def foobar():
        for i in range(10000):
            w.log2local('Line {}\n\n'.format(i))
            # time.sleep(.1)
    w.local_cancelButton.clicked.connect(foobar)
    w.setWindowTitle("Test this thing")
    w.show()
    sys.exit(app.exec_())
