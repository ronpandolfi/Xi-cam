import hipies
import sys, os

sys.path.append(os.path.join(os.getcwd(), 'lib/python2.7/lib-dynload'))
for path in sys.path:
    print 'path:', path

window = hipies.hipies.MyMainWindow()
