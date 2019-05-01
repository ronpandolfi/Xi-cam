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
        self.port = 27017

        from databroker import Broker

        # TODO: implement this case when the name of the .yml file is passed
        # db = Broker.named(str(host).lower())

        db_config = {'description': 'Configuration to access the data from mongodb',
                     'metadatastore': {
                         'module': 'databroker.headersource.mongo',
                         'class': 'MDS',
                         'config': {
                             'host': self.host,
                             'port': self.port,
                             'database': config.settings['Databroker MetaDataStore Name'],
                             'timezone': 'US/Eastern'}},
                     'assets': {
                         'module': 'databroker.assets.mongo',
                         'class': 'Registry',
                         'config': {
                             'host': self.host,
                             'port': self.port,
                             'database': config.settings['Databroker FileStore Name']}}
                     }

        db = Broker.from_config(db_config)
        self.db = db

    def __getitem__(self, item):
        return self.db[item]

    def __getattr__(self, item):
        return self.db.__getattr__(item)

    def __call__(self, *args, **kwargs):
        return self.db(*args, **kwargs)


class DBError(Exception):
    pass
