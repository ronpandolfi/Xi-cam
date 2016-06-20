import threads
import client

# Some HPC host addresses
HPC_SYSTEM_ADDRESSES = {'Cori': 'cori.nersc.gov', 'Edison': 'edison.nersc.gov', 'Bragg': 'bragg.dhcp.lbl.gov'}

# Clients and what not
# bind classes to new names
sftp_client = client.sftp.SFTPClient
globus_client = client.globus.GlobusClient

# Singleton instance of spot_client
spot_client = client.spot.SpotClient()

# Dicts to hold client instances
sftp_clients = {}
globus_clients = {}


def login_wrapper(client_login, *args, **kwargs):
    def handled_login(*args, **kwargs):
        try:
            return client_login(*args, **kwargs)
        except client.EXCEPTIONS as e:
            print e.message
            return
    return handled_login

def login(client_callback, client_login, credentials):
    handled_login = login_wrapper(client_login)
    runnable = threads.RunnableMethod(handled_login, method_kwargs=credentials, callback_slot=client_callback)
    threads.add_to_queue(runnable)


def add_sftp_client(host, client, callback):
    sftp_clients[host] = client
    callback(client)

def add_globus_client(endpoint, client, callback):
    globus_clients[endpoint] = client
    callback(client)

def logout(client_obj, callback=None):
    # TODO remove clients from their corresponding dictionaries!!!
    if hasattr(client_obj, 'logout'):
        method = client_obj.logout
    elif hasattr(client_obj, 'close'):
        method = client_obj.close
    runnable = threads.RunnableMethod(method, callback_slot=callback)
    threads.add_to_queue(runnable)


#TODO implement this to save NIM credentials
class NIMCredentials(object):
    """Class to save NIM user credentials to avoid inputting them soooo many times"""
    # Is this not secure? I am mangling the names though...
    def __init__(self, user, password):
        self.__user = user
        self.__password = password
