try:
    from tomocam import tomoCam
except ImportError:
    pass

def recon(tomo, theta, center=None, algorithm=None, input_params=None, **kwargs):

    if 'gridrec' in algorithm:
        return tomoCam.gpuGridrec(tomo, theta, center, input_params)
    elif 'sirt' in algorithm:
        return tomoCam.gpuSIRT(tomo, theta, center, input_params)
    else:
        raise ValueError('TomoCam reconstruction error')
