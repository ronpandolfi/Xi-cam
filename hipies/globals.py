import multiprocessing
from collections import OrderedDict

LUT = None
LUTlevels = None
LUTstate = None
plugins = OrderedDict()
pool = None

def load():
    global pool
    multiprocessing.freeze_support()
    pool = multiprocessing.Pool()
