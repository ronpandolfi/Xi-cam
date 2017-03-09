from .. import base
import widgets


import numpy as np



def runtest():
    EZTest.bottomwidget.clear()

def openfiles(filepaths):
    # handle new file
    pass

EZTest=base.EZplugin(name='HiTp',
                     toolbuttons=[],#('xicam/gui/icons_34.png',runtest)
                     parameters=[{'name':'Pre-edge Min','value':10,'type':'int'},
                                 {'name':'Pre-edge Max','value':30,'type':'int'}],
                     openfileshandler=openfiles,
                     centerwidget=widgets.WaferView,
                     bottomwidget=widgets.LocalView)