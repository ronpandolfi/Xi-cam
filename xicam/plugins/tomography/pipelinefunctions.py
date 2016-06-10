import numpy as np
# from skimage.exposure import rescale_intensity

DTYPE_RANGE = {'uint8': (0, 255),
               'uint16': (0, 65535),
               'int8': (-128, 127),
               'int16': (-32768, 32767),
               'float32': (-1, 1),
               'float64': (-1, 1)}

def crop(arr, p11, p12, p21, p22, axis=0):
    """
    Crops a 3D array along a given axis. Equivalent to slicing the array
    :param arr: ndarray
    :param p11: int, first point along first axis
    :param p12: int, second point along first axis
    :param p21: int, first point along second axis
    :param p22: int, second point along second axis
    :param axis: int, axis to crop along
    :return: ndarray, cropped array
    """
    slc = []
    pts = [p11, p12, p21, p22]
    for n in range(len(arr.shape)):
        if n == axis:
            slc.append(slice(None))
        else:
            slc.append(slice(pts.pop(0), -pts.pop(0)))
    return arr[slc]


def convert_data(arr, imin=None, imax=None, dtype='uint8'):
    """
    Convert an image or 3D array to another datatype
    :param arr: ndarray, data array
    :param dtype: dataype keyword
    :param imin,
    :param imax,
    :return: ndarry, converted to dtype
    """

    allowed_dtypes = ('uint8', 'uint16', 'int8', 'int16', 'float32', 'float64')
    if dtype not in allowed_dtypes:
        raise ValueError('dtype keyword {0} not in allowed keywords {1}'.format(dtype, allowed_dtypes))

    # Determine range to cast values
    if imin is None:
        imin = np.min(arr)
    if imax is None:
        imax = np.max(arr)

    # Determine range of new dtype
    omin, omax = DTYPE_RANGE[dtype]
    omin = 0 if imin >= 0 else omin

    # rescale pixel intensity values
    arr = np.clip(arr, imin, imax)
    arr = np.array(((arr - imin) / float(imax - imin))* (omax - omin) + omin, dtype=arr.dtype)
    # arr = rescale_intensity(arr, in_range=in_range, out_range=dtype)

    # Cast data to specified type
    return np.array(arr, dtype=np.dtype(dtype))


def array_operation(arr, value, operation='divide'):
    if operation not in ('add', 'subtract', 'multiply', 'divide'):
        raise ValueError('Operation {} is not a valid array operation'.format(operation))
    elif operation == 'add':
        return arr + value
    elif operation == 'subtract':
        return arr - value
    elif operation == 'multiply':
        return arr*value
    elif operation == 'divide':
        return arr/value


if __name__ == '__main__':
    import tomopy
    from matplotlib.pyplot import imshow, show, figure
    d = tomopy.read_als_832h5('/home/lbluque/TestDatasetsLocal/dleucopodia.h5', ind_tomo=(500, 501, 502, 503))
    c = convert_data(d[0], imin=0, imax=0)
    print d[0].dtype
    print c.dtype
    figure(0)
    imshow(d[0][0])
    figure(1)
    imshow(c[0])
    show()