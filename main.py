import hipies
import sys
import os
from PySide import QtGui
import multiprocessing

outfile = open(os.path.join(os.path.expanduser('~'),'error.log'), 'w')
sys.stdout = outfile
sys.stderr = outfile

if __name__=='__main__':
    multiprocessing.freeze_support()
    sys.path.append(os.path.join(os.getcwd(), 'lib/python2.7/lib-dynload'))
    for path in sys.path:
        print 'path:', path
    app=QtGui.QApplication(sys.argv)
    window = hipies.hipies.MyMainWindow(app)
    sys.exit(app.exec_())
