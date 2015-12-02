from PySide import QtGui
import hipies
import sys
import os
from hipies import globals


if __name__ == '__main__':
    globals.load()
    sys.path.append(os.path.join(os.getcwd(), 'lib/python2.7/lib-dynload'))
    for path in sys.path:
        print 'path:', path
    app=QtGui.QApplication(sys.argv)
    window = hipies.hipies.MyMainWindow(app)
    sys.exit(app.exec_())
