functionManifest = """
Center Finding:
    - displayName:  Fourier Autocorrelation
      functionName: FourierAutocorrelationCenterFind
      moduleName:   saxsfunctions
    - displayName:  Ricker Wavelets
      functionName: RickerWaveletCenterFind
      moduleName:   saxsfunctions
      parameters:
          - name:   Search minimum
            type:   int
            limits: [1,100000]
            suffix: ' px'
          - name:   Search maximum
            type:   int
            limits: [1,100000]
            suffix: ' px'

"""


def FourierAutocorrelationCenterFind(**workspace):
    from pipeline.center_approx import center_approx
    from xicam import config

    rawdata = workspace['rawdata']

    center = center_approx.center_approx(rawdata)

    config.activeExperiment.center = center
    workspace['center'] = center
    return workspace


def RickerWaveletCenterFind(minr, maxr, **workspace):
    import numpy as np
    from pipeline import center_approx
    from scipy import signal
    from xicam import config

    rawdata = workspace['rawdata']

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
    workspace['center'] = center
    return workspace
