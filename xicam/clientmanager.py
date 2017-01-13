

__author__ = "Luis Barroso-Luque"
__copyright__ = "Copyright 2016, CAMERA, LBL, ALS"
__credits__ = ["Ronald J Pandolfi", "Dinesh Kumar", "Singanallur Venkatakrishnan", "Luis Luque", "Alexander Hexemer"]
__license__ = ""
__version__ = "1.2.1"
__maintainer__ = "Ronald J Pandolfi"
__email__ = "ronpandolfi@lbl.gov"
__status__ = "Beta"


import threads
import client
from pipeline import msg

# Some default HPC host addresses
HPC_SYSTEM_ADDRESSES = {'Cori': 'cori.nersc.gov', 'Edison': 'edison.nersc.gov', 'Bragg': 'bragg.dhcp.lbl.gov'}

# Clients and what not
# bind classes to new names
sftp_client = client.sftp.SFTPClient
globus_client = client.globus.GlobusClient
ssh_client = client.ssh.SSHClient

# Singleton instance of spot_client
spot_client = client.spot.SpotClient()

# Dicts to hold client instances
sftp_clients = {}
globus_clients = {}
ssh_clients = {}


def login_wrapper(client_login):
    """Decorator to catch all login errors from NEWT, Globus, and PySFTP/Paramiko"""
    def handled_login(*args, **kwargs):
        try:
            return client_login(*args, **kwargs)
        except client.EXCEPTIONS as e:
            msg.logMessage(e.message,msg.ERROR)
            return

    return handled_login


def login(client_callback, client_login, credentials):
    """Login clients on a background thread"""
    handled_login = login_wrapper(client_login)
    bg_handled_login = threads.method(callback_slot=client_callback)(handled_login)
    bg_handled_login(**credentials)


def add_sftp_client(host, client, callback):
    """Add sftp client to dictionary in order to have them accessible to plugins"""
    sftp_clients[host] = client
    callback(client)


def add_ssh_client(host, client, callback):
    """Add sftp client to dictionary in order to have them accessible to plugins"""
    ssh_clients[host] = client
    callback(client)


def add_globus_client(endpoint, client, callback):
    """Add globus client to dictionary in order to have them accessible to plugins"""
    globus_clients[endpoint] = client
    callback(client)


def logout(client_obj, callback=None):
    """Logout client on a background thread"""
    # TODO remove clients from their corresponding dictionaries!!!
    if hasattr(client_obj, 'logout'):
        method = client_obj.logout
    elif hasattr(client_obj, 'close'):
        method = client_obj.close
    threads.method(callback_slot=callback)(method)()  # Decorate and run client logout/close method


# TODO implement this to save NIM credentials
class NIMCredentials(object):
    """Class to save NIM user credentials to avoid inputting them soooo many times"""
    # Is this not secure? I am mangling the names though...
    def __init__(self, user, password):
        self.__user = user
        self.__password = password
