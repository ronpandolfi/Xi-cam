import imp
try:
    imp.find_module("libcudart.so.7.5")
    found = True
except ImportError:
    found = False

if found:
    from tomocam import tomoCam
>>>>>>> 3e2aa18e431a5e8c1fd5d75edb262e8426090ffd

def recon(tomo, theta, center=None, algorithm=None, input_params=None, **kwargs):

    if 'gridrec' in algorithm:
        return tomoCam.gpuGridrec(tomo, theta, center, input_params)
    elif 'sirt' in algorithm:
        return tomoCam.gpuSIRT(tomo, theta, center, input_params)
    else:
        raise ValueError('TomoCam reconstruction error')
