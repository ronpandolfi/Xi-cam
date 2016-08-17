# -*- coding: UTF-8 -*-
import time
import inspect
import pyqtgraph as pg
import numpy as np
from pipeline import msg

def timeit(method):
    """
    Use this as a decorator to time a function
    """

    def timed(*args, **kw):
        curframe = inspect.currentframe()
        calframe = inspect.getouterframes(curframe, 2)
        # print 'Find peaks called from ' + calframe[1][3]
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        # '%r (%r, %r) %2.2f sec'
        #(method.__name__, args, kw, te - ts)
        msg.logMessage('%r  %2.3f sec' % \
              (method.__name__, te - ts),msg.DEBUG)
        return result

    return timed


def frustration():
    msg.logMessage(u"(ﾉಥ益ಥ)ﾉ﻿ ┻━┻",msg.CRITICAL)


def showimage(img):
    image = pg.image(np.fliplr(img))
