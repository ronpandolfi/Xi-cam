import multiprocessing
from collections import OrderedDict
from functools import partial
import threads
import client

LUT = None
LUTlevels = None
LUTstate = None
plugins = OrderedDict()
pool = None
window = None
lastroi = None
statusbar = None
app = None

def load():
    global pool
    multiprocessing.freeze_support()
    pool = multiprocessing.Pool()


def hardresetpool():
    global pool
    pool.terminate()
    pool.join()
    pool = multiprocessing.Pool()


# Login global variables, functions, and such
spot_client = client.spot.SpotClient()
globus_client = client.globus.GlobusClient()
login_callback = None
client_callback = None


def client_callback(*args, **kwargs):
    pass


def login_exeption_handle(func, *args, **kwargs):
    def handled_func(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except NotLoggedInError:
            return
    return(handled_func)


def load_spot(callback_func, *callback_args, **callback_kwargs):
    global spot_client, client_callback
    if not spot_client.logged_in:
        client_callback = partial(callback_func, *callback_args, **callback_kwargs)
        raise NotLoggedInError
    return spot_client


def login(client_obj, credentials):
    usr, pwd = map(str, credentials)
    global client_callback, login_callback
    if not client_obj.logged_in:
        runnable = threads.RunnableMethod(client_obj.login, method_args=(usr, pwd), callback_slot=client_callback)
        threads.add_to_queue(runnable)


def logout(client_obj, callback):
    runnable = threads.RunnableMethod(client_obj.logout, callback_slot=callback)
    threads.add_to_queue(runnable)


class NotLoggedInError(Exception):
    pass