

class DataBrokerClient(object): # replace with databroker client
    def __init__(self, host, **kwargs):
        import os
        import numpy as np
        from suitcase.als import ALSHDF5Handler, ALSHDF5SinoHandler
        super(DataBrokerClient, self).__init__()
        self.host = host

        # set up filestore

        from filestore.utils import install_sentinels
        from filestore.fs import FileStore
        from metadatastore.mds import MDS
        from databroker import Broker

        fs_config = {'host': 'localhost', 'port': 27017, 'database': 'fs_dev'}

        try:
            # this only needs to be done once
            install_sentinels(fs_config, 1)
        except RuntimeError:
            pass

        fs = FileStore(fs_config, version=1)

        fs.register_handler('ALS_HDF', ALSHDF5Handler)
        fs.register_handler('ALS_HDF_SINO', ALSHDF5SinoHandler)
        mds_conf = dict(database='mds_dev6', host='localhost',
                        port=27017, timezone='US/Eastern')

        mds = MDS(mds_conf, 1, auth=False)

        db = Broker(mds, fs)
        print(db)
        self.db = db

    def __getitem__(self, item):
        return self.db[item]

    def __getattr__(self, item):
        return self.db.__getattr__(item)

    def __call__(self, *args, **kwargs):
        return self.db(*args, **kwargs)


class DBError(Exception):
    pass
