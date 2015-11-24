import multiprocessing
from collections import OrderedDict

LUT = None
LUTlevels = None
LUTstate = None
plugins = OrderedDict()

pool = multiprocessing.Pool()