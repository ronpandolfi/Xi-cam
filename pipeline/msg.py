import logging
import inspect
import sys
import time

statusbar = None # Must be registered to output to a ui status bar

stdch = logging.StreamHandler(sys.stdout)

guilogcallable = None

DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL

def showMessage(s,timeout=0):
    if statusbar is not None:
        statusbar.showMessage(s,timeout)

    logMessage(s)

def logMessage(s,level=INFO,loggername=None):
    # ATTENTION: loggername is 'intelligently' determined with inspect. You probably want to leave it None.
    if loggername is not None:
        loggername = inspect.stack()[1][3]
    logger = logging.getLogger(loggername)
    stdch.setLevel(level)
    logger.addHandler(stdch)

    s = time.asctime()+'\t'+unicode(s)

    logger.log(level,s)
    if guilogcallable:
        guilogcallable(level,s)
    print s

def clearMessage():
    statusbar.clearMessage()

