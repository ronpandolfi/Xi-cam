# -*- coding: utf-8 -*-

import h5py


def read_als_832h5_metadata(fname):
    """
    Read metadata in ALS 8.3.2 hdf5 dataset files

    :param fname: str, Path to hdf5 file.
    :return dict: dictionary of metadata items
    """

    gdata = {}
    with h5py.File(fname, 'r') as f:

        g = _find_dataset_group(f)
        for key in g.attrs.keys():
            gdata[key] = g.attrs[key]

    return gdata


def _find_dataset_group(h5object):
    """
    Finds the group name containing the stack of projections datasets within
    a ALS BL8.3.2 hdf5 file
    """
    # Only one root key means only one dataset in BL8.3.2 current format
    keys = h5object.keys()
    if len(keys) == 1:
        if isinstance(h5object[keys[0]], h5py.Group):
            group_keys = h5object[keys[0]].keys()
            if isinstance(h5object[keys[0]][group_keys[0]], h5py.Dataset):
                return h5object[keys[0]]
            else:
                return _find_dataset_group(h5object[keys[0]])
        else:
            raise Exception('Unable to find dataset group')
    else:
        raise Exception('Unable to find dataset group')