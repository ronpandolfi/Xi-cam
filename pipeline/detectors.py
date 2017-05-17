from __future__ import unicode_literals
from __future__ import division
from builtins import zip
from past.utils import old_div
import pyFAI.detectors
import fabio
import logging
from collections import OrderedDict

class PrincetonMTE(pyFAI.detectors.Detector):
    """
    Princeton Instrument, PI-MTE CCD
    """
    force_pixel = True
    MAX_SHAPE = (2048, 2048)
    BINNED_PIXEL_SIZE = {1: 13.5e-6,
                         2: 27e-6}


    def __init__(self):
        pixel1 = 27e-6
        pixel2 = 27e-6
        super(PrincetonMTE, self).__init__(pixel1=pixel1, pixel2=pixel2)

    def get_binning(self):
        return self._binning

    def set_binning(self, bin_size=(1, 1)):
        """
        Set the "binning" of the detector,
        @param bin_size: set the binning of the detector
        @type bin_size: int or (int, int)
        """
        if "__len__" in dir(bin_size) and len(bin_size) >= 2:
            bin_size = int(round(float(bin_size[0]))), int(round(float(bin_size[1])))
        else:
            b = int(round(float(bin_size)))
            bin_size = (b, b)
        if bin_size != self._binning:
            if (bin_size[0] in self.BINNED_PIXEL_SIZE) and (bin_size[1] in self.BINNED_PIXEL_SIZE):
                self._pixel1 = self.BINNED_PIXEL_SIZE[bin_size[0]]
                self._pixel2 = self.BINNED_PIXEL_SIZE[bin_size[1]]
            else:
                # logger.warning("Binning factor (%sx%s) is not an official value for Princeton Instruments detectors" % (bin_size[0], bin_size[1]))
                self._pixel1 = old_div(self.BINNED_PIXEL_SIZE[1], float(bin_size[0]))
                self._pixel2 = old_div(self.BINNED_PIXEL_SIZE[1], float(bin_size[1]))
            self._binning = bin_size
            self.shape = (self.max_shape[0] // bin_size[0],
                          self.max_shape[1] // bin_size[1])

    binning = property(get_binning, set_binning)

    def __repr__(self):
        return "Detector %s\t PixelSize= %.3e, %.3e m" % \
               (self.name, self._pixel1, self._pixel2)

    def guess_binning(self, data):
        """
        Guess the binning/mode depending on the image shape
        @param data: 2-tuple with the shape of the image or the image with a .shape attribute.
        """
        if "shape" in dir(data):
            shape = data.shape
        else:
            shape = tuple(data[:2])
        bin1 = self.MAX_SHAPE[0] // shape[0]
        bin2 = self.MAX_SHAPE[1] // shape[1]
        self._binning = (bin1, bin2)
        self.shape = shape
        self.max_shape = shape
        self._pixel1 = self.BINNED_PIXEL_SIZE[bin1]
        self._pixel2 = self.BINNED_PIXEL_SIZE[bin2]
        self._mask = False
        self._mask_crc = None


class PhotonicScience(pyFAI.detectors.Detector):
    """
    Photonic Science CCD Detector

    """
    aliases = ["Photonic Science"]
    force_pixel = True
    MAX_SHAPE = (1042, 1042)
    DEFAULT_PIXEL1 = DEFAULT_PIXEL2 = 200e-6

    def __init__(self, pixel1=200e-6, pixel2=200e-6):
        super(PhotonicScience, self).__init__(pixel1=pixel1, pixel2=pixel2)
        if (pixel1 != self.DEFAULT_PIXEL1) or (pixel2 != self.DEFAULT_PIXEL2):
            self._binning = (int(2 * pixel1 / self.DEFAULT_PIXEL1), int(2 * pixel2 / self.DEFAULT_PIXEL2))
            self.shape = tuple(s // b for s, b in zip(self.max_shape, self._binning))
        else:
            self.shape = (old_div(1042,2), old_div(1042,2))
            self._binning = (2, 2)

    def __repr__(self):
        return "Detector %s\t PixelSize= %.3e, %.3e m" % \
            (self.name, self._pixel1, self._pixel2)

ALL_DETECTORS = OrderedDict((cls().name,cls) for cls in sorted(list(pyFAI.detectors.ALL_DETECTORS.values()),key=lambda cls:cls().name))