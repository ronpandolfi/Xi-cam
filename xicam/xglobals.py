from pathos import multiprocessing
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


def client_callback(*args, **kwargs):
    pass


def login_exeption_handle(func, *args, **kwargs):
    def handled_func(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except NotLoggedInError:
            return
    return(handled_func)


def load_spot(callback_func=client_callback, *callback_args, **callback_kwargs):
    global spot_client, client_callback
    if not spot_client.logged_in:
        client_callback = partial(callback_func, *callback_args, **callback_kwargs)
        raise NotLoggedInError
    return spot_client


def login(client, credentials):
    usr, pwd = map(str, credentials)
    global client_callback, login_callback
    if not spot_client.logged_in:
        runnable = threads.RunnableMethod(client_callback, client.login, usr, pwd)
        threads.queue.put(runnable)
        threads.worker_thread.start()


def logout(client, callback):
    runnable = threads.RunnableMethod(callback, client.logout)
    threads.queue.put(runnable)
    threads.worker_thread.start()


class NotLoggedInError(Exception):
    pass