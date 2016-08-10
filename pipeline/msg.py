import logging
import inspect
import sys
import time

statusbar = None # Must be registered to output to a ui status bar

stdch = logging.StreamHandler(sys.stdout)

guilogcallable = None

DEBUG = logging.DEBUG           # 10
INFO = logging.INFO             # 20
WARNING = logging.WARNING       # 30
ERROR = logging.ERROR           # 40
CRITICAL = logging.CRITICAL     # 50



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

    timestamp = time.asctime()
    m = timestamp +'\t'+unicode(s)

    logger.log(level,m)
    if guilogcallable:
        guilogcallable(level,timestamp,s)
    print m

def clearMessage():
    statusbar.clearMessage()

