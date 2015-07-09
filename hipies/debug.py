# -*- coding: UTF-8 -*-
import time


def timeit(method):
    """
    Use this as a decorator to time a function
    """

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        # '%r (%r, %r) %2.2f sec'
        #(method.__name__, args, kw, te - ts)
        print '%r  %2.3f sec' % \
              (method.__name__, te - ts)
        return result

    return timed


def frustration():
    print "(ﾉಥ益ಥ）ﾉ﻿ ┻━┻"