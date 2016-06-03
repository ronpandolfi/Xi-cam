import logging
import inspect

statusbar = None # Must be registered to output to a ui status bar



def showMessage(s,timeout=0):
    if statusbar is not None:
        statusbar.showMessage(s,timeout)
    else:
        logMessage(s)

def logMessage(s,level='info',loggername=None):
    # ATTENTION: loggername is 'intelligently' determined with inspect. You probably want to leave it None.
    if loggername is not None:
        loggername = inspect.stack()[1][3]
    logger = logging.getLogger(loggername)
    logger.log(level,s)

def clearMessage():
    statusbar.clearMessage()