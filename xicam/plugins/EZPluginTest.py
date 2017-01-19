import base

def runtest():
    import numpy as np

    img = np.random.random((100,100,100))
    EZTest.setImage(img)

    hist = np.histogram(img,100)
    EZTest.plot(hist[1][:-1],hist[0])

def opentest(filepaths):
    import fabio
    for filepath in filepaths:
        img = fabio.open(filepath).data
        EZTest.setImage(img)

EZTest=base.EZplugin(name='EZTest',toolbuttons=[('xicam/gui/icons_34.png',runtest)],parameters=[{'name':'Test','value':10,'type':'int'}],openfileshandler=opentest)

