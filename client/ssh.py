import paramiko


class SSHClient(paramiko.SSHClient):
    """Save a Paramiko SSH Connection"""

    def __init__(self, username,host,password):
        port = 22
        self.host = host

        super(SSHClient, self).__init__()
        self.load_system_host_keys()
        self.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.connect(host, port, username, password)

