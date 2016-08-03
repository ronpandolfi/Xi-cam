import pysftp


class SFTPClient(pysftp.Connection):
    """Save the host name of a pysftp Connection"""

    def __init__(self, host, **kwargs):
        super(SFTPClient, self).__init__(host, **kwargs)
        self.host = host
