import fabio

import pyfits
import os

def loadpath(path):
    """
    :type path : str
    :param path:
    :return:
    """

    if path.split('.')[-1] in '.fits .edf .tif':
        if path.split('.')[-1] == 'fits':
            data = pyfits.open(path)[2].data
        else:
            data = fabio.open(path).data

        txtpath = '.'.join(path.split('.')[:-1]) + '.txt'
        # print txtpath
        if os.path.isfile(txtpath):
            with open(txtpath, 'r') as f:
                lines = f.readlines()
                paras = dict()
                i = 0
                for line in lines:
                    cells = line.split(':')
                    if cells.__len__() == 2:
                        paras[cells[0]] = float(cells[1])
                    elif cells.__len__() == 1:
                        i += 1
                        paras['Unknown' + str(i)] = str(cells[0])
            print paras
            return data, paras
        else:
            return data, None

    return None, None

    # except TypeError:
    #   print('Failed to load',path,', its probably not an image format I understand.')
    #  return None,None