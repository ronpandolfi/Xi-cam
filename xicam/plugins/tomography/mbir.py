# this is supposed to prevent tomocam import if it's not installed
# doesn't work: will exclude tomocam even with correct installations
import imp
try:
    imp.find_module("libcudart.so.7.5")
    found = True
except ImportError:
    found = False

if found:
    from tomocam import tomoCam

def recon(tomo, theta, center=None, algorithm=None, input_params=None, **kwargs):

    if 'gridrec' in algorithm:
        return tomoCam.gpuGridrec(tomo, theta, center, input_params)
    elif 'sirt' in algorithm:
        return tomoCam.gpuSIRT(tomo, theta, center, input_params)
    else:
        raise ValueError('TomoCam reconstruction error')
