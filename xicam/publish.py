from PySide.QtCore import *
from PySide.QtGui import *
from PySide.QtUiTools import *

def publish():
    wizard = PublishWizard()
    wizard.exec_()

class PublishWizard(QWizard):
    def __new__(cls, *args, **kwargs):
        """Load widget from UI file."""
        super(PublishWizard, cls).__new__(cls, *args, **kwargs)
        uifile = QFile('xicam/gui/publish.ui')
        uifile.open(QFile.ReadOnly)
        wizard = QUiLoader().load(uifile)
        uifile.close()
        return wizard

class PublishPackage(object):
    def __init__(self):
        self.license =
    def validate(self):
        return True