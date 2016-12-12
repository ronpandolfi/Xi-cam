from PySide.QtGui import *
import editor
import console

class advancedPythonWidget(QTabWidget):
    def __init__(self):
        super(advancedPythonWidget, self).__init__()

        self.ipythonconsole=console.ipythonconsole()
        self.scripteditor=editor.scripteditor()

        self.addTab(self.ipythonconsole,'IPython')
        self.addTab(self.scripteditor,'Script Editor')

    def __getattr__(self, attr):  ## implicitly wrap methods from children
        for widget in [self.ipythonconsole,self.scripteditor]:
            if hasattr(widget, attr):
                m = getattr(widget, attr)
                if hasattr(m, '__call__'):
                    return m
        raise NameError(attr)