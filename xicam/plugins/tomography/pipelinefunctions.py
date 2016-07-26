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


def convert_data(arr, imin=None, imax=None, dtype='uint8', intcast='float32'):
    """
    Convert an image or 3D array to another datatype

    :param arr: ndarray, data array
    :param dtype: dataype keyword
    :param imin,
    :param imax,
    :param intcast: datatype to cast ints to
    :return: ndarry, converted to dtype
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



def array_operation(arr, value, operation='divide'):
    """
    Simple array operations

    :param arr:
    :param value:
    :param operation:
    :return:
    """
    if operation not in ('add', 'subtract', 'multiply', 'divide'):
        raise ValueError('Operation {} is not a valid array operation'.format(operation))
    elif operation == 'add':
        return ne.evaluate('arr + value')
    elif operation == 'subtract':
        return ne.evaluate('arr - value', truediv=True)
    elif operation == 'multiply':
        return ne.evaluate('arr * value')
    elif operation == 'divide':
        return ne.evaluate('arr / value')


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