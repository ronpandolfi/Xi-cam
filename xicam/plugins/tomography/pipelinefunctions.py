

__author__ = "Luis Barroso-Luque"
__copyright__ = "Copyright 2016, CAMERA, LBL, ALS"
__credits__ = ["Ronald J Pandolfi", "Dinesh Kumar", "Singanallur Venkatakrishnan", "Luis Luque", "Alexander Hexemer"]
__license__ = ""
__version__ = "1.2.1"
__maintainer__ = "Ronald J Pandolfi"
__email__ = "ronpandolfi@lbl.gov"
__status__ = "Beta"


import numpy as np
import numexpr as ne

DTYPE_RANGE = {'uint8': (0, 255),
               'uint16': (0, 65535),
               'int8': (-128, 127),
               'int16': (-32768, 32767),
               'float32': (-1, 1),
               'float64': (-1, 1)}

def crop(arr, p11, p12, p21, p22, axis=0):
    """
    Crops a 3D array along a given axis. Equivalent to slicing the array

    Parameters
    ----------
    arr : ndarray
    p11 : int
        First point along first axis
    p12 : int
        Second point along first axis
    p21 : int
        First point along second axis
    p22 : int
        Second point along second axis
    axis : int
        Axis to crop along

    Returns
    -------
    ndarray:
        Cropped array
    """

    slc = []
    pts = [p11, p12, p21, p22]
    for n in range(len(arr.shape)):
        if n == axis:
            slc.append(slice(None))
        else:
            slc.append(slice(pts.pop(0), -pts.pop(0)))
    return arr[slc]


def convert_data(arr, imin=None, imax=None, dtype='uint8', intcast='float32'):
    """
    Convert an image or 3D array to another datatype using numexpr

    :param arr: ndarray, data array
    :param dtype: dataype keyword
    :param imin,
    :param imax,
    :param intcast: datatype to cast ints to
    :return: ndarry, converted to dtype

    Parameters
    ----------
    arr : ndarray, data array
    imin : int/float, optional
        Minimum bound of input array. If not given the minimum value of the array is used
    imax : int/float, optional
        Maximum bound of input array. If not given the maximum value of the array is used
    dtype : str
        Dataype keyword. See DTYPE_RANGE keys
    intcast : str
        intermediate cast type if casting to ints (numexpr will complain otherwise)

    Returns
    -------
    ndarray
        Converted array
    """

    allowed_dtypes = ('uint8', 'uint16', 'int8', 'int16', 'float32', 'float64')
    if dtype not in allowed_dtypes:
        raise ValueError('dtype keyword {0} not in allowed keywords {1}'.format(dtype, allowed_dtypes))

    # Determine range to cast values
    minset=False
    if imin is None:
        imin = np.min(arr)
        minset=True
    maxset=False
    if imax is None:
        imax = np.max(arr)
        maxset=True

    np_cast = getattr(np, str(arr.dtype))
    imin, imax =  np_cast(imin),  np_cast(imax)

    # Determine range of new dtype
    omin, omax = DTYPE_RANGE[dtype]
    omin = 0 if imin >= 0 else omin
    omin, omax = np_cast(omin), np_cast(omax)

    if arr.dtype in [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16,
                     np.uint32, np.uint64, np.bool_, np.int_, np.intc, np.intp]:
        int_cast = getattr(np, str(intcast))
        out = np.empty(arr.shape, dtype=int_cast)
        imin = int_cast(imin)
        imax = int_cast(imax)
        df = int_cast(imax) - int_cast(imin)
    else:
        out = np.empty(arr.shape, dtype=arr.dtype)
        df = imax - imin
    if not minset:
        if np.min(arr) < imin:
            arr = ne.evaluate('where(arr < imin, imin, arr)', out=out)
    if not maxset:
        if np.max(arr) > imax:
            arr = ne.evaluate('where(arr > imax, imax, arr)', out=out)
    ne.evaluate('(arr - imin) / df', truediv=True, out=out)
    ne.evaluate("out * (omax - omin) + omin", out=out)

    # Cast data to specified type
    return out.astype(np.dtype(dtype), copy=False)


# series of wrappers for simple array operations to expose in workflow pipeline GUI
def array_operation_add(arr, value=0):
    return ne.evaluate('arr + value')

def array_operation_sub(arr, value=0):
    return ne.evaluate('arr - value', truediv=True)

def array_operation_mult(arr, value=1):
    return ne.evaluate('arr * value')

def array_operation_div(arr, value=1):
    return ne.evaluate('arr / value')

def array_operation_max(arr, value=0):
    return np.maximum(arr, value)

def reader(start_sinogram=0, end_sinogram=0, step_sinogram=1, start_projection=0, end_projection=0, step_projection=1,
           sinograms_per_chunk=0):

    """
    Function to expose input tomography array parameters in function pipeline GUI

    Parameters
    ----------
    start_sinogram : int
        Start for sinogram processing
    end_sinogram : int
        End for sinogram processing
    step_sinogram : int
        Step for sinogram processing
    start_projection : int
        Start for projection processing
    end_projection : int
        End for projection processing
    step_projection : int
        Step for projection processing
    sinograms_per_chunk : int
        Number of sinograms processed at one time. Limited by machine memory size

    Returns
    -------
    tuple, tuple, int:
        Arguments rearranged into tuples to feed into reconstruction
    """

    return (start_projection, end_projection, step_projection), (start_sinogram, end_sinogram, step_sinogram), \
           sinograms_per_chunk

def slicer(arr, p11=0, p12=0, p21=0, p22=0, p31=0, p32=0):
    """
    Slices a 3D array according the points given

    Parameters
    ----------
    arr : ndarray
    p11 : int
        First point along first axis
    p12 : int
        Second point along first axis
    p21 : int
        First point along second axis
    p22 : int
        Second point along second axis
    p31 : int
        First point along third axis
    p32 : int
        Second point along third axis
    axis : int
        Axis to crop along

    Returns
    -------
    ndarray:
        Cropped array
    """

    slc = []
    pts = [p11, p12, p21, p22, p31, p32]
    for n in range(len(arr.shape)):
        slc.append(slice(pts.pop(0), pts.pop(0)))
    return arr[slc]



if __name__ == '__main__':
    import tomopy
    from matplotlib.pyplot import imshow, show, figure
    d = tomopy.read_als_832h5('/home/lbluque/TestDatasetsLocal/dleucopodia.h5', ind_tomo=(500, 501, 502, 503))
    # d = np.array(d[0], dtype=np.float32)
    d = d[0]
    c = convert_data(d, imin=0, imax=0)
    print d.dtype
    print c.dtype
    figure(0)
    imshow(d[0])
    figure(1)
    imshow(c[0])
    show()