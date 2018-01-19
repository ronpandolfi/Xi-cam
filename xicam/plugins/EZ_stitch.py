import base
import numpy as np
import pyFAI
import fabio
def stitching(data1, data2):

    # DEFAULT TILING AT 733
    positionY1 = 25 * .172
    positionY2 = 0
    positionX1 = 25 * .172
    positionX2 = 0

    I1 = 1
    I2 = 1

    deltaX = round((positionX2 - positionX1) / 0.172)
    deltaY = round((positionY2 - positionY1) / 0.172)
    padtop2 = 0
    padbottom1 = 0
    padtop1 = 0
    padbottom2 = 0
    padleft2 = 0
    padright1 = 0
    padleft1 = 0
    padright2 = 0

    if deltaY < 0:
        padtop2 = int(abs(deltaY))
        padbottom1 = int(abs(deltaY))
    else:
        padtop1 = int(abs(deltaY))
        padbottom2 = int(abs(deltaY))

    if deltaX < 0:
        padleft2 = int(abs(deltaX))
        padright1 = int(abs(deltaX))

    else:
        padleft1 = int(abs(deltaX))
        padright2 = int(abs(deltaX))

    d2 = np.pad(data2, ((padtop2, padbottom2), (padleft2, padright2)), 'constant')
    d1 = np.pad(data1, ((padtop1, padbottom1), (padleft1, padright1)), 'constant')

    mask2 = np.pad(1 - pyFAI.detectors.Pilatus2M().calc_mask(),
                      ((padtop2, padbottom2), (padleft2, padright2)),
                      'constant')
    mask1 = np.pad(1 - pyFAI.detectors.Pilatus2M().calc_mask(),
                      ((padtop1, padbottom1), (padleft1, padright1)),
                      'constant')

    with np.errstate(divide='ignore',invalid='ignore'):
        data = (d1 / I1 + d2 / I2) / (mask2 + mask1) * (I1 + I2) / 2.
        data[np.isnan(data)] = 0
    return data, np.logical_or(mask2, mask1).astype(np.int)

def opentest(filepaths):
    print(filepaths[0], filepaths[1])
    im0 = fabio.open(filepaths[0]).data
    im1 = fabio.open(filepaths[1]).data
    img_st, test = stitching(im0, im1)
    EZTest.setImage((np.rot90(img_st, 1) + 0.01) )


EZTest=base.EZplugin(name='stitching',toolbuttons=[('xicam/gui/icons_28.png',stitching)],
                     parameters=[{'name':'Pitch','value':2200,'type':'int'},
                                 {'name':'Radius', 'value': 600, 'type': 'int'},
                                 {'name':'Shell', 'value': 98, 'type': 'int'},
                                 {'name':'Coherence', 'value': 2, 'type': 'int'},
                                 {'name':'Linewidth', 'value': 1000, 'type': 'int'}
                                 ],openfileshandler=opentest)


