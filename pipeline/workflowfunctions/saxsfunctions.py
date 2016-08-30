from pipeline.workflowfunctions import updateworkspace

functionManifest = """
Center Finding:
    - displayName:  Fourier Autocorrelation
      functionName: FourierAutocorrelationCenterFind
      moduleName:   saxsfunctions
      functionType: PROCESS
    - displayName:  Ricker Wavelets
      functionName: RickerWaveletCenterFind
      moduleName:   saxsfunctions
      functionType: PROCESS
      parameters:
          - name:   Search minimum
            type:   int
            limits: [1,100000]
            suffix: ' px'
          - name:   Search maximum
            type:   int
            limits: [1,100000]
            suffix: ' px'
Integration:
    - displayName:  Radial Integrate
      functionName: RadialIntegrate
      moduleName:   saxsfunctions
      functionType: PROCESS
      parameters:
        - name:     Bins
          type:     int
          limits:   [1,100000]
Write to File:
    - displayName:  Write to CSV
      functionName: WriteCSV
      moduleName:   saxsfunctions
      functionType: OUTPUT
      parameters:
        - name:     File suffix
          type:     str
          value:    '_reduced'


"""


def FourierAutocorrelationCenterFind(**workspace):
    from pipeline.center_approx import center_approx
    from xicam import config

    rawdata = workspace['dimg'].rawdata

    center = center_approx(rawdata)

    config.activeExperiment.center = center
    updates = {'center': center}
    updateworkspace(workspace, updates)
    return workspace, updates


def RickerWaveletCenterFind(minr, maxr, **workspace):
    import numpy as np
    from pipeline import center_approx
    from scipy import signal
    from xicam import config

    rawdata = workspace['dimg'].rawdata

    radii = np.arange(minr, maxr)
    maxval = 0
    center = np.array([0, 0], dtype=np.int)
    for i in range(len(radii)):
        w = center_approx.tophat2(radii[i], scale=1000)
        im2 = signal.fftconvolve(rawdata, w, 'same')
        if im2.max() > maxval:
            maxval = im2.max()
            center = np.array(np.unravel_index(im2.argmax(), rawdata.shape))

    config.activeExperiment.center = center
    updates = {'center': center}
    updateworkspace(workspace, updates)
    return workspace, updates


def RadialIntegrate(bins, **workspace):
    from pipeline import integration

    data = workspace['dimg'].transformdata

    q, radialprofile, _, _ = integration.qintegrate(data, **workspace)

    updates = {'radialintegration': [q, radialprofile]}
    updateworkspace(workspace, updates)
    return workspace, updates


def WriteCSV(suffix, updates, **workspace):
    from pipeline.writer import writearray

    path = workspace['dimg'].filepath

    for key, value in updates.iteritems():
        writearray(value, path, suffix=suffix)
