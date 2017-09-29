import pyFAI
import numpy

import logging
logger = logging.getLogger("pyFAI.azimuthalIntegrator")


# monkey patch to correct auto-inversion of masks
class AzimuthalIntegrator(pyFAI.AzimuthalIntegrator):


    def create_mask(self, data, mask=None,
                    dummy=None, delta_dummy=None, mode="normal"):
        """
        Combines various masks into another one.

        @param data: input array of data
        @type data: ndarray
        @param mask: input mask (if none, self.mask is used)
        @type mask: ndarray
        @param dummy: value of dead pixels
        @type dummy: float
        @param delta_dumy: precision of dummy pixels
        @type delta_dummy: float
        @param mode: can be "normal" or "numpy" (inverted) or "where" applied to the mask
        @type mode: str

        @return: the new mask
        @rtype: ndarray of bool

        This method combine two masks (dynamic mask from *data &
        dummy* and *mask*) to generate a new one with the 'or' binary
        operation.  One can adjust the level, with the *dummy* and
        the *delta_dummy* parameter, when you consider the *data*
        values needs to be masked out.

        This method can work in two different *mode*:

            * "normal": False for valid pixels, True for bad pixels
            * "numpy": True for valid pixels, false for others

        This method tries to accomodate various types of masks (like
        valid=0 & masked=-1, ...) and guesses if an input mask needs
        to be inverted.
        """
        shape = data.shape
        #       ^^^^   this is why data is mandatory !
        if mask is None:
            mask = self.mask
        if mask is None:
            mask = numpy.zeros(shape, dtype=bool)
        elif mask.min() < 0 and mask.max() == 0:  # 0 is valid, <0 is invalid
            mask = (mask < 0)
        else:
            mask = mask.astype(bool)
        # if mask.sum(dtype=int) > mask.size // 2:                              # TERRRIBLEE!
        #     logger.warning("Mask likely to be inverted as more"
        #                    " than half pixel are masked !!!")
        #     numpy.logical_not(mask, mask)
        if (mask.shape != shape):
            try:
                mask = mask[:shape[0], :shape[1]]
            except Exception as error:  # IGNORE:W0703
                logger.error("Mask provided has wrong shape:"
                             " expected: %s, got %s, error: %s" %
                             (shape, mask.shape, error))
                mask = numpy.zeros(shape, dtype=bool)
        if dummy is not None:
            if delta_dummy is None:
                numpy.logical_or(mask, (data == dummy), mask)
            else:
                numpy.logical_or(mask,
                                 abs(data - dummy) <= delta_dummy,
                                 mask)
        if mode == "numpy":
            numpy.logical_not(mask, mask)
        elif mode == "where":
            mask = numpy.where(numpy.logical_not(mask))
        return mask

    def __getnewargs_ex__(self):
        # TODO: also allow detector object to be pickled (currently they are recycled as None)
        return self.dist, self.poni1, self.poni2,self.rot1, self.rot2, self.rot3,self.pixel1, self.pixel2,self.splineFile, None, self.wavelength

    def __getstate__(self):
        return (self._flatfield, 
                self._darkcurrent,
                self._flatfield_crc,
                self._darkcurrent_crc,
                self._writer,
                self.flatfiles,
                self.darkfiles,
                self.header,
                self._ocl_integrator,
                self._ocl_lut_integr,
                self._ocl_csr_integr,
                self._lut_integrator,
                self._csr_integrator,
                self._ocl_sorter)

    def __setstate__(self, state):
        self._flatfield, self._darkcurrent,self._flatfield_crc,self._darkcurrent_crc,self._writer,self.flatfiles, self.darkfiles,self.header,self._ocl_integrator,self._ocl_lut_integr,self._ocl_csr_integr,self._lut_integrator,self._csr_integrator,self._ocl_sorter = state

pyFAI.__dict__['AzimuthalIntegrator']=AzimuthalIntegrator

def test_AzimuthalIntegrator_pickle():
    import pickle
    import numpy as np
    det=pyFAI.detectors.detector_factory('pilatus2m')
    ai=AzimuthalIntegrator(detector=det)
    ai.set_wavelength(.1)
    ai.integrate1d(np.zeros(det.shape),1000) # force lut generation
    print(ai.__reduce__())
    print(pickle.dumps(ai))
