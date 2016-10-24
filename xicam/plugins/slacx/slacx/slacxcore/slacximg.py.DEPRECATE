import os
import re

from PIL import Image, ImageQt
import fabio
import numpy as np

from . import slacxex

class SlacxImage(object):
    """
    Container class for slacx images.
    """

    tifftest = re.compile("^.tif*$")
    martest = re.compile("^.mar.*$")

    def __init__(self,img_url):
        self.img_url = img_url
        #self.img_filename = img_url.split('/')[-1]
        #self.img_name = os.path.splitext(self.img_filename)[0]
        #self.img_ext = os.path.splitext(self.img_filename)[1]
        #self.rendered = False
        self.img_data = None 
        #self.img_hdr = None 
        #self.pil_img = None
        #self.fab_img = None

    def istiff(self):
        img_ext = os.path.splitext(self.img_url)[1]
        if self.tifftest.match(img_ext):
            return True
        return False

    def size_tag(self):
        if self.img_data is not None:
            imgsize = self.img_data.shape
            sz_tag = '{} by {} array'.format(imgsize[0],imgsize[1])
            return sz_tag
        else:
            return 'no image data'

    def load_img_data(self):
        """
        Call on image rendering libs to extract image data
        *.tiff or *.tif images: use PIL
        *.mar* images: use fabio?
        *.raw images: ??
        """
        # TODO: make sure to only do this if it has not already been done
        if self.img_data is None: 
            if self.istiff():
                try:
                    # Open the tif, convert it to grayscale with .convert("L")
                    pil_img = Image.open(self.img_url).convert("L")
                    # Get the data out of this image.
                    # Image.getdata() returns a sequence (have to reshape):
                    self.img_data = np.array(pil_img.getdata()).reshape(pil_img.size).T
                    #self.rendered = True
                    #import pdb; pdb.set_trace()
                    #print 'Image data attributes...'
                    #print 'shape: {}'.format(self.img_data.shape)
                    #print 'maxval: {}'.format(np.max(self.img_data))
                    #print 'minval: {}'.format(np.min(self.img_data))
                except IOError:
                    print "[{}] PIL IOError for file {}".format(
                    __name__,self.img_url)
                    raise
            else:
                msg = '[{}] TODO: image data loading for files like {}'.format(
                __name__,self.img_url)
                raise slacxex.LazyCodeError(msg)

    #def close(self):
    #    pass
    #    # TODO: Self-destruct all data 

    #### ARCHIVE ####
    #def qt_image(self):
    #    if self.pil_img:
    #        try:
    #            # ImageQt.ImageQt(Image) is a subclass of QtGui.QImage.
    #            # This works, but the result segfaults when converted to QPixmap?
    #            return ImageQt.ImageQt(self.pil_img)
    #        except:
    #            print "[{}] PIL error in ImageQt conversion for {}".format(
    #            __name__,self.img_filename)
    #            raise
    #    else:
    #        msg = '[{}] TODO: handle QtImage conversion for files like {}'.format(
    #            __name__,self.img_filename)
    #        raise slacxex.LazyCodeError(msg)

