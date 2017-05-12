from PIL import Image
import numpy as np


def load_image(imageFullname):
    # open MARCCD tiff image
    im = Image.open(imageFullname)
    # input image object into an array
    imArray = np.array(im)
    im.close()
    return imArray