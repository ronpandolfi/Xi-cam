"""
author: Fang Ren (SSRL)

4/26/2017
"""
import fabio
import numpy as np

def load_image(imageFullname):
    # open MARCCD tiff image
    im = fabio.open(imageFullname)
    # input image object into an array
    imArray = im.data
    return imArray