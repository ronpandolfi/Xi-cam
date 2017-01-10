import logging
import inspect
import sys
import time

statusbar = None # Must be registered to output to a ui status bar
progressbar = None

stdch = logging.StreamHandler(sys.stdout)

guilogcallable = None

DEBUG = logging.DEBUG           # 10
INFO = logging.INFO             # 20
WARNING = logging.WARNING       # 30
ERROR = logging.ERROR           # 40
CRITICAL = logging.CRITICAL     # 50

logbacklog=[]


def showProgress(value, min=0, max=100):
    if progressbar:
        progressbar.show()
        progressbar.setRange(min, max)
        progressbar.setValue(value)


def showBusy():
    if progressbar:
        progressbar.show()
        progressbar.setRange(0, 0)


def hideBusy():
    if progressbar:
        progressbar.hide()
        progressbar.setRange(0, 100)


hideProgress = hideBusy


def showMessage(s,timeout=0):
    if statusbar is not None:
        statusbar.showMessage(s, timeout * 1000)

    logMessage(s)

def logMessage(stuple,level=INFO,loggername=None,timestamp=None,image=None,suppressreprint=False):

    # ATTENTION: loggername is 'intelligently' determined with inspect. You probably want to leave it None.
    if loggername is not None:
        loggername = inspect.stack()[1][3]
    logger = logging.getLogger(loggername)
    try:
        stdch.setLevel(level)
    except ValueError:
        print stuple,level
    logger.addHandler(stdch)

    if timestamp is None: timestamp = time.asctime()

    if type(stuple) is not tuple:
        stuple=[stuple]

    stuple = (unicode(s) for s in stuple)

    s = ' '.join(stuple)
    m = timestamp +'\t'+unicode(s)

    logger.log(level,m)
    if guilogcallable:
        guilogcallable(level,timestamp,s,image)
    else:
        global logbacklog
        logbacklog.append({'stuple':s,'level':level,'loggername':loggername,'timestamp':timestamp,'image':image})
    try:
        if not suppressreprint: print m
    except UnicodeEncodeError:
        print 'A unicode string could not be written to console. Some logging will not be displayed.'

def flushbacklog():
    global logbacklog
    for l in logbacklog:
        l['suppressreprint']=True
        logMessage(**l)
    logbacklog=[]

def clearMessage():
    statusbar.clearMessage()

