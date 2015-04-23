import fabio

import pyfits


def loadpath(path):
    """
    :type path : str
    :param path:
    :return:
    """
    if path.split('.')[-1] == 'fits':
        return pyfits.open(path)[2].data
    else:
        return fabio.open(path).data