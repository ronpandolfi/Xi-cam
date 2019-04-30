from xicam import config


class DataBrokerClient(object):  # replace with databroker client
    def __init__(self, host, **kwargs):
        import os
        import numpy as np
        # TODO: handlers will need to be registered
        #
        # from suitcase.als832 import ALSHDF5Handler, ALSHDF5SinoHandler
        # from suitcase.als733 import ALSEDFHandler
        super(DataBrokerClient, self).__init__()
        self.host = host

        # set up filestore

        # from filestore.utils import install_sentinels
        # from filestore.fs import FileStore
        from databroker.assets.mongo import Registry
        from databroker.headersource.mongo import MDS
        from databroker import Broker
        # from databroker.core import register_builtin_handlers

        fs_config = {'host': host, 'port': 27017,
                     'database': config.settings['Databroker FileStore Name']}

        # try:
        #     # this only needs to be done once
        #     install_sentinels(fs_config, 1)
        # except RuntimeError:
        #     pass

        fs = Registry(fs_config, version=1)

        # fs.register_handler('ALS_HDF', ALSHDF5Handler)
        # fs.register_handler('ALS_HDF_SINO', ALSHDF5SinoHandler)
        # fs.register_handler('ALS_EDF', ALSEDFHandler)
        # register_builtin_handlers(fs)
        mds_conf = dict(database=config.settings['Databroker MetaDataStore Name'], host=host,
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
