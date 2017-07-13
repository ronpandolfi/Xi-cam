

__author__ = "Luis Barroso-Luque, Holden Parks"
__copyright__ = "Copyright 2016, CAMERA, LBL, ALS"
__credits__ = ["Ronald J Pandolfi", "Dinesh Kumar", "Singanallur Venkatakrishnan", "Luis Luque",
               "Holden Parks", "Alexander Hexemer"]
__license__ = ""
__version__ = "1.2.1"
__maintainer__ = "Ronald J Pandolfi"
__email__ = "ronpandolfi@lbl.gov"
__status__ = "Beta"


import numpy as np
import numexpr as ne
import concurrent.futures as cf
import tomopy
from tomopy.util import mproc
import scipy.ndimage.filters as snf
import skimage.transform as st


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
           sinograms_per_chunk=0, projections_per_chunk=0):

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
           sinograms_per_chunk, projections_per_chunk

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


def remove_outlier1d(arr, dif, size=3, axis=0, ncore=None, out=None):
    """
    Remove high intensity bright spots (and dark spots) from an array, using a one-dimensional
    median filter along the specified axis

    Parameters
    ----------
    arr : ndarray
        Input array.
    dif : float
        Expected difference value between outlier value and
        the median value of the array.
    size : int
        Size of the median filter.
    axis : int, optional
        Axis along which median filtering is performed.
    ncore : int, optional
        Number of cores that will be assigned to jobs.
    out : ndarray, optional
        Output array for result.  If same as arr, process will be done in-place.
    Returns
    -------
    ndarray
       Corrected array.
    """
    arr = arr.astype(np.float32, copy=False)
    dif = np.float32(dif)

    tmp = np.empty_like(arr)

    other_axes = [i for i in range(arr.ndim) if i != axis]
    largest = np.argmax([arr.shape[i] for i in other_axes])
    lar_axis = other_axes[largest]
    ncore, chnk_slices = mproc.get_ncore_slices(arr.shape[lar_axis], ncore=ncore)
    filt_size = [1] * arr.ndim
    filt_size[axis] = size

    with cf.ThreadPoolExecutor(ncore) as e:
        slc = [slice(None)] * arr.ndim
        for i in range(ncore):
            slc[lar_axis] = chnk_slices[i]
            e.submit(snf.median_filter, arr[slc], size=filt_size, output=tmp[slc], mode='mirror')

    with mproc.set_numexpr_threads(ncore):
        out = ne.evaluate('where(abs(arr-tmp)>=dif,tmp,arr)', out=out)

    return out

def beam_hardening(arr, a0=0, a1=1.0, a2=0, a3=0, a4=0, a5=0.1):
    """
    beam hardening correction, based on "Correction for beam hardening in computed tomography",
    Gabor Herman, 1979 Phys. Med. Biol. 24 81

    Correction is: tomo = a0 + a1*tomo + a2*tomo^2 + a3*tomo^3 + a4*tomo^4 + a5*tomo^5
    """

    loc_dict = {}
    loc_dict['a0'] = np.float32(a0)
    loc_dict['a1'] = np.float32(a1)
    loc_dict['a2'] = np.float32(a2)
    loc_dict['a3'] = np.float32(a3)
    loc_dict['a4'] = np.float32(a4)
    loc_dict['a5'] = np.float32(a5)

    return ne.evaluate('a0 + a1*tomo + a2*tomo**2 + a3*tomo**3 + a4*tomo**4 + a5*tomo**5', local_dict=loc_dict)

def correct_tilt(arr, tilt=0, tiltcenter_slice=None, tiltcenter_det=None, sino_0=0):
    """
    Offset dataset tilt

    Parameters
    ----------
    arr : ndarray
        Input array.
    tilt :
    tiltcenter_slice : int, optional
        Center of dataset in x (sinogram) direction
    tiltcenter_det: int, optional
        Center of dataset in y (image height) direction
    sino_0 : int, optional
        Position of first sinogram in 'arr' relative to larger dataset. For example, if sino_0=200, then the first
        sinogram in 'arr' is the 200th in the larger dataset from which 'arr' is derived
    Returns
    -------
    ndarray
       Corrected array.
    """

    if not tiltcenter_slice:
        tiltcenter_slice = arr.shape[1]/2
    if not tiltcenter_det:
        tiltcenter_det = arr.shape[2]/2

    new_center = tiltcenter_slice - 0.5 -sino_0
    center_det = tiltcenter_det - 0.5
    cntr = (center_det, new_center)
    for b in range(arr.shape[0]):
        arr[b] = st.rotate(arr[b], tilt, center=cntr, preserve_range=True, order=1, mode='edge', clip=True)

    return arr

def sino_360_to_180(data, overlap=0, rotation='left'):
    """
    Wrapper for 360_to_180 function (see below) to handle even/odd shaped data
    """

    if data.shape[0]%2>0:
        return do360_to_180(data[0:-1,:,:], overlap=overlap, rotation=rotation)
    else:
        return do360_to_180(data[:,:,:], overlap=overlap, rotation=rotation)


def do360_to_180(data, overlap=0, rotation='left'):
    """
    Converts 0-360 degrees sinogram to a 0-180 sinogram.

    Parameters
    ----------
    data : ndarray
        Input 3D data.

    overlap : scalar, optional
        Overlapping number of pixels.

    rotation : string, optional
        Left if rotation center is close to the left of the
        field-of-view, right otherwise.

    Returns
    -------
    ndarray
    Output 3D data.
    """
    dx, dy, dz = data.shape
    lo = overlap // 2
    ro = overlap - lo
    n = dx // 2
    out = np.zeros((n, dy, 2*dz - overlap), dtype=data.dtype)
    if rotation == 'left':
        weights = (np.arange(overlap) + 0.5) / overlap
        out[:, :, -dz + overlap:] = data[:n, :, overlap:]
        out[:, :, :dz - overlap] = data[n:2 * n, :, overlap:][:, :, ::-1]
        out[:, :, dz - overlap:dz] = weights * data[:n, :, :overlap] + (weights * data[n:2 * n, :, :overlap])[:, :,
                                                                       ::-1]
    elif rotation == 'right':
        weights = (np.arange(overlap)[::-1] + 0.5) / overlap
        out[:, :, :dz - overlap] = data[:n, :, :-overlap]
        out[:, :, -dz + overlap:] = data[n:2 * n, :, :-overlap][:, :, ::-1]
        out[:, :, dz - overlap:dz] = weights * data[:n, :, -overlap:] + (weights * data[n:2 * n, :, -overlap:])[:, :,
                                                                        ::-1]
    return out

def normalize(tomo, flats, dark, cutoff=None, ncore=None):
    """
    Wrapper for tomopy.normalize.normalize function (make more similar to nearest flats norm function)
    """

    return tomopy.normalize(arr=tomo, flat=flats, dark=dark, cutoff=cutoff, ncore=ncore)


if __name__ == '__main__':
    import tomopy
    from matplotlib.pyplot import imshow, show, figure
    d = tomopy.read_als_832h5('/home/lbluque/TestDatasetsLocal/dleucopodia.h5', ind_tomo=(500, 501, 502, 503))
    # d = np.array(d[0], dtype=np.float32)
    d = d[0]
    c = convert_data(d, imin=0, imax=0)
    print(d.dtype)
    print(c.dtype)
    figure(0)
    imshow(d[0])
    figure(1)
    imshow(c[0])
    show()