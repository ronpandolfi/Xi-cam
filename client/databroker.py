
class DataBrokerClient(object): # replace with databroker client
    def __init__(self, host, **kwargs):
        super(DataBrokerClient, self).__init__()
        self.host=host
        print(host)


class DBError(Exception):
    pass