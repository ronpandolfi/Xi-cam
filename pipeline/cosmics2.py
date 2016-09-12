"""
Remove cosmic rays / x-ray strikes from image data.

About
=====

cosmics.py is a small and simple python module to detect and clean cosmic ray hits on images (numpy arrays or FITS),
using scipy, and based on Pieter van Dokkum's L.A.Cosmic algorithm.  Pythonized by Malte Tewes;
further alterations by Amanda Fournier.

L.A.Cosmic = Laplacian cosmic ray detection

U{http://www.astro.yale.edu/dokkum/lacosmic/}

(article : U{http://arxiv.org/abs/astro-ph/0108003})


Additional features
===================

I pimped this a bit to suit my needs :

	- Automatic recognition of saturated stars, including their full saturation trails.
	This avoids that such stars are treated as big cosmics.
	Indeed saturated stars tend to get even uglier when you try to clean them. Plus they
	keep L.A.Cosmic iterations going on forever.
	This feature is mainly for pretty-image production. It is optional, requires one more parameter (a CCD saturation level in ADU), and uses some 
	nicely robust morphology operations and object extraction.
	
	- Scipy image analysis allows to "label" the actual cosmic ray hits (i.e. group the pixels into local islands).
	A bit special, but I use this in the scope of visualizing a PSF construction.

But otherwise the core is really a 1-to-1 implementation of L.A.Cosmic, and uses the same parameters.
Only the conventions on how filters are applied at the image edges might be different.

No surprise, this python module is much faster then the IRAF implementation, as it does not read/write every step to disk.

Usage
=====

Everything is in the file cosmics.py, all you need to do is to import it. You need pyfits, numpy and scipy.
See the demo scripts for example usages (the second demo uses f2n.py to make pngs, and thus also needs PIL).

Your image should have clean borders, cut away prescan/overscan etc.



Todo
====
Ideas for future improvements :

	- Add something reliable to detect negative glitches (dust on CCD or small traps)
	- Top level functions to simply run all this on either numpy arrays or directly on FITS files
	- Reduce memory usage ... easy
	- Switch from signal to ndimage, homogenize mirror boundaries


Malte Tewes, January 2010
"""

__version__ = '0.4'

import os
import numpy as np
import scipy.signal as signal
import scipy.ndimage as ndimage
import pyfits

# TODO: time initial algorithm to establish baseline speed
# TODO: initial mask is a must-have.
# TODO: remove/replace all references to laplkernel and growkernel?
# TODO: subsampling is kind of bad, let's not subsample.
# TODO: localized successive iterations?
# TODO: time final algorithm
# TODO:


# We define the laplacian kernel to be used
# laplkernel = np.array([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]])
# Other kernels :
# growkernel = np.ones((3, 3))
dilstruct = np.ones((5, 5))  # dilation structure for some morphological operations
dilstruct[0, 0] = 0
dilstruct[0, -1] = 0
dilstruct[-1, 0] = 0
dilstruct[-1, -1] = 0


# So this dilstruct looks like :
# 01110
# 11111
# 11111
# 11111
# 01110
# and is used to dilate saturated stars and connect cosmic rays.


class cosmicsimage:
    def __init__(self, rawarray, premask=np.zeros(1), pssl=0.0, gain=2.2, readnoise=10.0, sigclip=5.0, sigfrac=0.3,
                 objlim=5.0,
                 satlevel=50000.0, verbose=True):
        """

        sigclip : increase this if you detect cosmics where there are none. Default is 5.0, a good value for earth-bound images.
        objlim : increase this if normal stars are detected as cosmics. Default is 5.0, a good value for earth-bound images.

        Constructor of the cosmic class, takes a 2D numpy array of your image as main argument.
        sigclip : laplacian-to-noise limit for cosmic ray detection
        objlim : minimum contrast between laplacian image and fine structure image. Use 5.0 if your image is undersampled, HST, ...

        satlevel : if we find agglomerations of pixels above this level, we consider it to be a saturated star and
        do not try to correct and pixels around it. A negative satlevel skips this feature.

        pssl is the previously subtracted sky level !

        real   gain    = 1.8          # gain (electrons/ADU)	(0=unknown)
        real   readn   = 6.5		      # read noise (electrons) (0=unknown)
#        ##gain0  string statsec = "*,*"       # section to use for automatic computation of gain
        real   skyval  = 0.           # sky level that has been subtracted (ADU)
        real   sigclip = 3.0          # detection limit for cosmic rays (sigma)
        real   sigfrac = 0.5          # fractional detection limit for neighbouring pixels
        real   objlim  = 3.0           # contrast limit between CR and underlying object
        int    niter   = 1            # maximum number of iterations

        """
        self.rawarray = rawarray + pssl  # internally, we work "with sky" because it affects noise statistics.
        self.cleanarray = self.rawarray.copy()  # In lacosmiciteration() we work on this guy
        if premask.any():
            self.premask = premask
        else:
            self.premask = np.zeros(rawarray.shape, dtype=bool)

        self.zingermask = np.zeros(rawarray.shape, dtype=bool)  # All False, no cosmics yet
        self.holemask = np.zeros(rawarray.shape, dtype=bool)  # a mask of 'negative cosmics', e.g. dust on detector
        self.satmask = np.zeros(rawarray.shape, dtype=bool)  # a mask of (not-to-be-corrected) saturated regions
        #        self.allnonfeaturemask = premask.copy() # (premask | satmask) = allnonfeaturemask
        #        self.allfeaturemask = np.zeros(rawarray.shape, dtype=bool) # (zingermask | holemask) = allfeaturemask

        self.gain = gain
        self.readnoise = readnoise
        self.sigclip = sigclip
        self.objlim = objlim
        self.sigcliplow = sigclip * sigfrac
        self.satlevel = satlevel

        self.verbose = verbose

        self.pssl = pssl

        self.backgroundlevel = None  # Only calculated/used if very large zingers present
        self.satstars = None  # Namespace detritus from old version

    def __str__(self):
        """
        Gives a summary of the current state, including the number of cosmic pixels in the mask etc.
        """
        stringlist = [
            "Input array : (%i, %i), %s" % (self.rawarray.shape[0], self.rawarray.shape[1], self.rawarray.dtype.name),
            "Current cosmic ray mask : %i pixels" % np.sum(self.mask)
        ]

        if self.pssl != 0.0:
            stringlist.append("Using a previously subtracted sky level of %f" % self.pssl)

        if self.satstars != None:
            stringlist.append("Saturated star mask : %i pixels" % np.sum(self.satstars))

        return "\n".join(stringlist)

    # Not clear why I'd want/need this function.
    '''
    def labelmask(self, verbose=None):
        """
        Finds and labels the cosmic "islands" and returns a list of dicts containing their positions.
        This is made on purpose for visualizations a la f2n.drawstarslist, but could be useful anyway.
        """
        if verbose == None:
            verbose = self.verbose
        if verbose:
            print "Labeling mask pixels ..."
        # We morphologicaly dilate the mask to generously connect "sparse" cosmics :
        #dilstruct = np.ones((5,5))
        dilmask = ndimage.morphology.binary_dilation(self.mask, structure=dilstruct, iterations=1, mask=None,
                                                     output=None, border_value=0, origin=0, brute_force=False)
        # origin = 0 means center
        (labels, n) = ndimage.measurements.label(dilmask)
        #print "Number of cosmic ray hits : %i" % n
        #tofits(labels, "labels.fits", verbose = False)
        slicecouplelist = ndimage.measurements.find_objects(labels)
        # Now we have a huge list of couples of numpy slice objects giving a frame around each object
        # For plotting purposes, we want to transform this into the center of each object.
        if len(slicecouplelist) != n:
            # This never happened, but you never know ...
            raise RuntimeError, "Mega error in labelmask !"
        centers = [[(tup[0].start + tup[0].stop) / 2.0, (tup[1].start + tup[1].stop) / 2.0] for tup in slicecouplelist]
        # We also want to know how many pixels where affected by each cosmic ray.
        # Why ? Dunno... it's fun and available in scipy :-)
        sizes = ndimage.measurements.sum(self.mask.ravel(), labels.ravel(), np.arange(1, n + 1, 1))
        retdictlist = [{"name": "%i" % size, "x": center[0], "y": center[1]} for (size, center) in zip(sizes, centers)]

        if verbose:
            print "Labeling done"

        return retdictlist
    '''

    def getdilatedmask(self, size=3):
        """
        Returns a morphologically dilated copy of the current mask.
        size = 3 or 5 decides how to dilate.
        """
        if size == 3:
            dilmask = ndimage.morphology.binary_dilation(self.mask, structure=growkernel, iterations=1, mask=None,
                                                         output=None, border_value=0, origin=0, brute_force=False)
        elif size == 5:
            dilmask = ndimage.morphology.binary_dilation(self.mask, structure=dilstruct, iterations=1, mask=None,
                                                         output=None, border_value=0, origin=0, brute_force=False)
        else:
            dilmask = self.mask.copy()  #### Surely not correct?
        return dilmask

    def clean(self, mask=None, verbose=None):  # thumbsup?
        """
        Given the mask, we replace the actual problematic pixels with the masked 5x5 median value.
        This mimics what is done in L.A.Cosmic, but it's a bit harder to do in python, as there is no
        readymade masked median. So for now we do a loop...
        Saturated stars, if calculated, are also masked : they are not "cleaned", but their pixels are not
        used for the interpolation.

        We will directly change self.cleanimage. Instead of using the self.mask, you can supply your
        own mask as argument. This might be useful to apply this cleaning function iteratively.
        But for the true L.A.Cosmic, we don't use this, i.e. we use the full mask at each iteration.

        """
        if verbose == None:
            verbose = self.verbose
        if mask == None:
            mask = (self.zingermask | self.holemask)
        nonfeaturemask = (self.satmask | self.premask)

        if verbose:
            print "Cleaning cosmic affected pixels ..."

        # So... mask is a 2D array containing False and True, where True means "here is a cosmic"

        # WHY THIS SECTION??? Isn't the whole point to leave saturated areas unaltered?
        '''
        # Now in this copy called cleancopy, we also put the saturated stars to np.Inf, if available :
        if self.satstars != None:
            cleancopy[self.satregions] = np.Inf
        '''

        neighborhood, shift_mask = square_stack(self.cleanarray, 2)
        all_mask = (self.premask | self.satmask | self.zingermask | self.holemask)
        all_mask_neighborhood, _ = square_stack(all_mask, 2)  # Do not use values from pixels masked for any reason
        all_mask_shift_mask = all_mask_neighborhood * shift_mask
        median, median_mask = masked_median_2d_axis_0(neighborhood, all_mask_shift_mask)
        median = median.reshape(self.cleanarray.shape)
        median_mask = median_mask.reshape(self.cleanarray.shape)  # True = undefined, False = defined

        needs_replaced = (self.zingermask | self.holemask)  # True = invalid/needs replaced, False = valid
        replaceable_mask = ((~median_mask) & needs_replaced)  # True = can be replaced with masked median
        irreplaceable_mask = (median_mask & needs_replaced)  # True = cannot be replaced with masked median

        self.cleanarray[replaceable_mask] = median[replaceable_mask]
        # Next clause triggers only if there are 5x5 or larger zingers or holes.
        if irreplaceable_mask.any():
            if verbose:
                print "Alarmingly messy, isn't it?"
            self.updatebackgroundlevel()
            self.cleanarray[irreplaceable_mask] = self.backgroundlevel

        # That's it.
        if verbose:
            print "Cleaning done"

    def findsatstars(self, verbose=None):
        """
        Uses the satlevel to find saturated stars (not cosmics !), and puts the result as a mask in self.satstars.
        This can then be used to avoid these regions in cosmic detection and cleaning procedures.
        Slow ...
        """
        if verbose == None:
            verbose = self.verbose
        if verbose:
            print "Detecting saturated stars ..."
        # DETECTION

        satpixels = self.rawarray > self.satlevel  # the candidate pixels

        # We build a smoothed version of the image to look for large stars and their support :
        m5 = ndimage.filters.median_filter(self.rawarray, size=5, mode='mirror')
        # We look where this is above half the satlevel
        largestruct = m5 > (self.satlevel / 2.0)
        # The rough locations of saturated stars are now :
        satstarscenters = np.logical_and(largestruct, satpixels)

        if verbose:
            print "Building mask of saturated stars ..."

        # BUILDING THE MASK
        # The subtility is that we want to include all saturated pixels connected to these saturated stars...
        # I haven't found a better solution then the double loop

        # We dilate the satpixels alone, to ensure connectivity in glitchy regions and to add a safety margin around them.
        # dilstruct = np.array([[0,1,0], [1,1,1], [0,1,0]])

        dilsatpixels = ndimage.morphology.binary_dilation(satpixels, structure=dilstruct, iterations=2, mask=None,
                                                          output=None, border_value=0, origin=0, brute_force=False)
        # It turns out it's better to think large and do 2 iterations...


        # We label these :
        (dilsatlabels, nsat) = ndimage.measurements.label(dilsatpixels)
        # tofits(dilsatlabels, "test.fits")

        if verbose:
            print "We have %i saturated stars." % nsat

        # The ouput, False for now :
        outmask = np.zeros(self.rawarray.shape)

        for i in range(1, nsat + 1):  # we go through the islands of saturated pixels
            thisisland = dilsatlabels == i  # gives us a boolean array
            # Does this intersect with satstarscenters ?
            overlap = np.logical_and(thisisland, satstarscenters)
            if np.sum(overlap) > 0:
                outmask = np.logical_or(outmask, thisisland)  # we add thisisland to the mask

        self.satstars = np.cast['bool'](outmask)

        if verbose:
            print "Mask of saturated stars done"

    '''
    def getsatstars(self, verbose=None):
        """
        Returns the mask of saturated stars after finding them if not yet done.
        Intended mainly for external use.
        """
        if verbose == None:
            verbose = self.verbose
        if not self.satlevel > 0:
            raise RuntimeError, "Cannot determine satstars : you gave satlevel <= 0 !"
        if self.satstars == None:
            self.findsatstars(verbose=verbose)
        return self.satstars
    '''

    '''
    def getmask(self):
        return self.mask
    '''

    def getrawarray(self):
        """
        For external use only, as it returns the rawarray minus pssl !
        """
        return self.rawarray - self.pssl

    def getcleanarray(self):
        """
        For external use only, as it returns the cleanarray minus pssl !
        """
        return self.cleanarray - self.pssl

    def updatebackgroundlevel(self):
        """
        Estimates the background level if neighborhood completely invalid.

        Takes the median of all pixels not invalidated by various effects.
        Will not be actively used unless the 5x5 neighborhood around a zinged pixel is entirely invalid.
        While replacement values for zinged pixels are always invalid as data,
        that goes double for pixels guessed by this function.
        """
        full_mask = (self.zingermask or self.holemask or self.premask or self.satmask)
        self.backgroundlevel = np.median(self.rawarray[~full_mask])

    def lacosmiciteration(self, verbose=None):
        """
        Performs one iteration of the L.A.Cosmic algorithm.
        It operates on self.cleanarray, and afterwards updates self.mask by adding the newly detected
        cosmics to the existing self.mask. Cleaning is not made automatically ! You have to call
        clean() after each iteration.
        This way you can run it several times in a row to to L.A.Cosmic "iterations".
        See function lacosmic, that mimics the full iterative L.A.Cosmic algorithm.

        Returns a dict containing
            - niter : the number of cosmic pixels detected in this iteration
            - nnew : among these, how many were not yet in the mask
            - itermask : the mask of pixels detected in this iteration
            - newmask : the pixels detected that were not yet in the mask

        If findsatstars() was called, we exclude these regions from the search.

        """

        if verbose == None:
            verbose = self.verbose

        if verbose:
            print "Convolving image with Laplacian kernel ..."

        full_mask = (self.premask | self.satmask | self.zingermask | self.holemask)
        lplus = positive_laplacian(self.rawarray, full_mask)

        if verbose:
            print "Creating noise model ..."

        # We build a custom noise map, so to compare the laplacian to
        m5 = ndimage.filters.median_filter(self.cleanarray, size=5, mode='mirror')
        # We keep this m5, as I will use it later for the interpolation.
        m5clipped = m5.clip(min=0)  # As we will take the sqrt
        noise = (self.gain ** -1) * np.sqrt(self.gain * m5clipped + self.readnoise ** 2)

        # d.clip(min=0) # 0.0203399658203 s
        # d[d<0] = 0    # 0.00575804710388 s
        # ndimage.filters.median_filter(d, size=5, mode='mirror')  # 1.53711891174 s

        '''
        import time
        d, dout, dmask = gen_test_arrays(1024, 0.1)
        t0 = time.time()
        # 2.35331201553 s - 4kx4k,3x3,0.2; 6.25063109398 s - 4kx4k,5x5,0.2;
        # 0.603759050369 s - 2kx2k,3x3,0.2; 1.64695119858 s - 2kx2k,5x5,0.2; 2.68022704124 s - 2kx2k,7x7,0.2;
        # 0.155918836594 s - 1kx1k,3x3,0.2; 0.705060005188 s - 1kx1k,5x5,0.2; 0.691349029541 s - 1kx1k,7x7,0.2; 1.06284308434 s - 1kx1k,9x9,0.2;
#        scipy_med_filter(d, dout, dmask, 1)
        # 149.271306038 s - 4kx4k,3x3,0.2; 152.090310812 s - 4kx4k,5x5,0.2;
        # 37.1048049927 s - 2kx2k,3x3,0.2; 37.4898169041 s - 2kx2k,5x5,0.2; 40.9122078419 s - 2kx2k,7x7,0.2;
        # 9.3817589283 s - 1kx1k,3x3,0.2; 9.73028206825 s - 1kx1k,5x5,0.2; 10.2621040344 s - 1kx1k,7x7,0.2; 10.1984000206 s - 1kx1k,9x9,0.2;
        # 5.16152405739 s - 1kx1k,3x3,0.1; 5.19565796852 s - 1kx1k,5x5,0.1; 5.34027600288 s - 1kx1k,7x7,0.1; 5.65969014168 s - 1kx1k,9x9,0.1;
#        old_med_filter(d, dout, dmask, 4)
#        # 42.9699928761 s - 512x512,3x3,0.2;
#        # 2.64555120468 s - 256x256,3x3,0.2;
#        # 0.346574068069 s - 128x128,3x3,0.2;
#        alt_old_med_filter(d, dout, dmask, 1) # mega sucks.  why?
        # 11.4415609837 s - 4kx4k,3x3,0.2; effing disaster - 4kx4k,5x5,0.2;
        # 2.68856716156 s - 2kx2k,3x3,0.2; 14.1315348148 s - 2kx2k,5x5,0.2; 51.4261059761 s - 2kx2k,7x7,0.2;
        # 0.70032119751 s - 1kx1k,3x3,0.2; 3.37346100807 s - 1kx1k,5x5,0.2; 9.608481884 s - 1kx1k,7x7,0.2; 24.066202879 s - 1kx1k,9x9,0.2;
        # 0.660758018494 s - 1kx1k,3x3,0.1; 3.35483193398 s - 1kx1k,5x5,0.1; 9.94300603867 s - 1kx1k,7x7,0.1; 23.1019711494 s - 1kx1k,9x9,0.1;
        new_med_filter(d, dout, dmask, 4)
        t1 = time.time()
        dt = t1 - t0
        print dt
        '''

        if verbose:
            print "Calculating Laplacian signal to noise ratio ..."

        # Laplacian signal to noise ratio :
        s = lplus / noise
        # This s is called sigmap in the original lacosmic.cl

        # We remove the large structures (s prime) :
        sp = s - ndimage.filters.median_filter(s, size=5, mode='mirror')

        if verbose:
            print "Selecting candidate cosmic rays ..."

        # Candidate cosmic rays (this will include stars + HII regions)
        candidates = sp > self.sigclip
        nbcandidates = np.sum(candidates)

        if verbose:
            print "  %5i candidate pixels" % nbcandidates

        # At this stage we use the saturated stars to mask the candidates, if available :
        if self.satstars != None:
            if verbose:
                print "Masking saturated stars ..."
            candidates = np.logical_and(np.logical_not(self.satstars), candidates)
            nbcandidates = np.sum(candidates)

            if verbose:
                print "  %5i candidate pixels not part of saturated stars" % nbcandidates

        if verbose:
            print "Building fine structure image ..."

        # We build the fine structure image :
        m3 = ndimage.filters.median_filter(self.cleanarray, size=3, mode='mirror')
        m37 = ndimage.filters.median_filter(m3, size=7, mode='mirror')
        f = m3 - m37
        # In the article that's it, but in lacosmic.cl f is divided by the noise...
        # Ok I understand why, it depends on if you use sp/f or L+/f as criterion.
        # There are some differences between the article and the iraf implementation.
        # So I will stick to the iraf implementation.
        f = f / noise
        f = f.clip(min=0.01)  # as we will divide by f. like in the iraf version.

        if verbose:
            print "Removing suspected compact bright objects ..."

        # Now we have our better selection of cosmics :
        cosmics = np.logical_and(candidates, sp / f > self.objlim)
        # Note the sp/f and not lplus/f ... due to the f = f/noise above.

        nbcosmics = np.sum(cosmics)

        if verbose:
            print "  %5i remaining candidate pixels" % nbcosmics

        # What follows is a special treatment for neighbors, with more relaxed constains.

        if verbose:
            print "Finding neighboring pixels affected by cosmic rays ..."

        # We grow these cosmics a first time to determine the immediate neighborhod  :
        growcosmics = grow_eight_directions(cosmics)  # 0.012 s, down from 0.216 s

        # From this grown set, we keep those that have sp > sigmalim
        # so obviously not requiring sp/f > objlim, otherwise it would be pointless
        growcosmics = np.logical_and(sp > self.sigclip, growcosmics)

        # Now we repeat this procedure, but lower the detection limit to sigmalimlow :

        finalsel = grow_eight_directions(growcosmics)
        finalsel = (sp > self.sigcliplow) & finalsel

        # Again, we have to kick out pixels on saturated stars :
        if self.satstars != None:
            if verbose:
                print "Masking saturated stars ..."
            finalsel = np.logical_and(np.logical_not(self.satstars), finalsel)

        nbfinal = np.sum(finalsel)

        if verbose:
            print "  %5i pixels detected as cosmics" % nbfinal

        # Now the replacement of the cosmics...
        # we outsource this to the function clean(), as for some purposes the cleaning might not even be needed.
        # Easy way without masking would be :
        # self.cleanarray[finalsel] = m5[finalsel]

        # We find how many cosmics are not yet known :
        newmask = np.logical_and(np.logical_not(self.zingermask), finalsel)
        nbnew = np.sum(newmask)

        # We update the mask with the cosmics we have found :
        self.zingermask = np.logical_or(self.zingermask, finalsel)

        # We return
        # (used by function lacosmic)

        return {"niter": nbfinal, "nnew": nbnew, "itermask": finalsel, "newmask": newmask}

    def findholes(self, verbose=True):
        """
        Detects "negative cosmics" in the cleanarray and adds them to the mask.
        This is not working yet.
        """
        pass

        """
        if verbose == None:
            verbose = self.verbose

        if verbose :
            print "Finding holes ..."

        m3 = ndimage.filters.median_filter(self.cleanarray, size=3, mode='mirror')
        h = (m3 - self.cleanarray).clip(min=0.0)

        tofits("h.fits", h)
        sys.exit()

        # The holes are the peaks in this image that are not stars

        #holes = h > 300
        """
        """
        subsam = subsample(self.cleanarray)
        conved = -signal.convolve2d(subsam, laplkernel, mode="same", boundary="symm")
        cliped = conved.clip(min=0.0)
        lplus = rebin2x2(conved)

        tofits("lplus.fits", lplus)

         m5 = ndimage.filters.median_filter(self.cleanarray, size=5, mode='mirror')
         m5clipped = m5.clip(min=0.00001)
         noise = (1.0/self.gain) * np.sqrt(self.gain*m5clipped + self.readnoise*self.readnoise)

         s = lplus / (2.0 * noise) # the 2.0 is from the 2x2 subsampling
         # This s is called sigmap in the original lacosmic.cl

         # We remove the large structures (s prime) :
         sp = s - ndimage.filters.median_filter(s, size=5, mode='mirror')

         holes = sp > self.sigclip
        """
        """
        # We have to kick out pixels on saturated stars :
        if self.satstars != None:
             if verbose:
                 print "Masking saturated stars ..."
             holes = np.logical_and(np.logical_not(self.satstars), holes)

        if verbose:
            print "%i hole pixels found" % np.sum(holes)

        # We update the mask with the holes we have found :
        self.mask = np.logical_or(self.mask, holes)
        """

    def run(self, maxiter=4, verbose=False):
        """
        Full artillery :-)
            - Find saturated stars
            - Run maxiter L.A.Cosmic iterations (stops if no more cosmics are found)

        Stops if no cosmics are found or if maxiter is reached.
        """

        if self.satlevel > 0 and self.satstars == None:
            self.findsatstars(verbose=True)

        print "Starting %i L.A.Cosmic iterations ..." % maxiter
        for i in range(1, maxiter + 1):
            print "Iteration %i" % i

            iterres = self.lacosmiciteration(verbose=verbose)
            print "%i cosmic pixels (%i new)" % (iterres["niter"], iterres["nnew"])

            # self.clean(mask = iterres["mask"]) # No, we want clean to operate on really clean pixels only !
            # Thus we always apply it on the full mask, as lacosmic does :
            self.clean(verbose=verbose)
            # But note that for huge cosmics, one might want to revise this.
            # Thats why I added a feature to skip saturated stars !

            if iterres["niter"] == 0:
                break
        # A concession to naming issues
        self.mask = self.zingermask


# Top-level functions


# def fullarray(verbose = False):
# 	"""
# 	Applies the full artillery using and returning only numpy arrays
# 	"""
# 	pass
# 
# def fullfits(infile, outcleanfile = None, outmaskfile = None):
# 	"""
# 	Applies the full artillery of the function fullarray() directly on FITS files.
# 	"""
# 	pass



# FITS import - export

def fromfits(infilename, hdu=0, verbose=True):
    """
    Reads a FITS file and returns a 2D numpy array of the data.
    Use hdu to specify which HDU you want (default = primary = 0)
    """

    pixelarray, hdr = pyfits.getdata(infilename, hdu, header=True)
    pixelarray = np.asarray(pixelarray).transpose()

    pixelarrayshape = pixelarray.shape
    if verbose:
        print "FITS import shape : (%i, %i)" % (pixelarrayshape[0], pixelarrayshape[1])
        print "FITS file BITPIX : %s" % (hdr["BITPIX"])
        print "Internal array type :", pixelarray.dtype.name

    return pixelarray, hdr


def tofits(outfilename, pixelarray, hdr=None, verbose=True):
    """
    Takes a 2D numpy array and write it into a FITS file.
    If you specify a header (pyfits format, as returned by fromfits()) it will be used for the image.
    You can give me boolean numpy arrays, I will convert them into 8 bit integers.
    """
    pixelarrayshape = pixelarray.shape
    if verbose:
        print "FITS export shape : (%i, %i)" % (pixelarrayshape[0], pixelarrayshape[1])

    if pixelarray.dtype.name == "bool":
        pixelarray = np.cast["uint8"](pixelarray)

    if os.path.isfile(outfilename):
        os.remove(outfilename)

    if hdr == None:  # then a minimal header will be created
        hdu = pyfits.PrimaryHDU(pixelarray.transpose())
    else:  # this if else is probably not needed but anyway ...
        hdu = pyfits.PrimaryHDU(pixelarray.transpose(), hdr)

    hdu.writeto(outfilename)

    if verbose:
        print "Wrote %s" % outfilename


# Array manipulation


def shift_stack(y, n1, n2):
    '''
    Creates a stack of index-shifted versions of y.

    :param y: 1d numpy float array
    :param n1: int
    :param n2: int
    :return local_neighborhood: 2d numpy float array
    :return unreliable: 2d numpy bool array

    Creates shifted versions of the input *y*,
    with shifts up to and including *n1* spaces downward in index
    and up to and including *n2* spaces upwards.
    The shifted versions are stacked together as *local_neighborhood*, like this
    (shown for a *y* of length 16 and *n1 = 4*, *n2 = 2*)
    [4 5 6 7 ... 15 __ __ __ __]
    [3 4 5 6 ... 14 15 __ __ __]
    [2 3 4 5 ... 13 14 15 __ __]
    [1 2 3 4 ... 12 13 14 15 __]
    [0 1 2 3 ... 11 12 13 14 15]
    [_ 0 1 2 ... 10 11 12 13 14]
    [_ _ 0 1 ...  9 10 11 12 13]
    with a corresponding mask array, *unreliable*,
    indicating whether an element holds credible information or not, like this
    [0 0 0 0 ...  0  1  1  1  1]
    [0 0 0 0 ...  0  0  1  1  1]
    [0 0 0 0 ...  0  0  0  1  1]
    [0 0 0 0 ...  0  0  0  0  1]
    [0 0 0 0 ...  0  0  0  0  0]
    [1 0 0 0 ...  0  0  0  0  0]
    [1 1 0 0 ...  0  0  0  0  0]
    False indicates unmasked (credible) data,
    True indicates masked (unreliable) data.
    '''
    local_neighborhood = np.inf * np.ones(((n1 + n2 + 1), y.size), dtype=float)
    unreliable = np.ones(((n1 + n2 + 1), y.size), dtype=bool)
    for ii in range(n1 + n2 + 1):
        # ii ranges from 0 to n1 + n2; jj ranges from -n1 to n2
        jj = ii - n1
        jj_l_s, jj_l_e, jj_r_s, jj_r_e = stack_slice_indices(jj)
        local_neighborhood[ii, jj_l_s:jj_l_e] = y[jj_r_s:jj_r_e]
        unreliable[ii, jj_l_s:jj_l_e] = False
    return local_neighborhood, unreliable


def square_stack(y, n):
    '''
    Creates a stack of index-shifted versions of y from n by n region around each pixel.

    :param y: 2d numpy float array
    :param ymask: 2d numpy bool array
    :param n: int
    :return neighborhood: 2d numpy float array
    :return neighbor_mask: 2d numpy bool array
    '''
    # Blank arrays to store outputs
    dtype = y.dtype
    (height, width) = y.shape
    new_width = height * width
    new_height = (2 * n + 1) ** 2
    new_shape = (new_height, new_width)
    if dtype == float:
        local_neighborhood = np.inf * np.ones(new_shape, dtype=dtype)
    elif dtype == bool:
        local_neighborhood = np.ones(new_shape, dtype=dtype)
    neighborhood_mask = np.ones(new_shape, dtype=bool)
    # Blank arrays to use in calculations
    floats_invalid = np.inf * np.ones(y.shape, dtype=float)
    #    bools_false = np.zeros(y.shape, dtype=bool)
    bools_true = np.ones(y.shape, dtype=bool)
    #
    for ii in range(-n, n + 1):
        for jj in range(-n, n + 1):
            # kk a unique index between 0 and (2n+1)**2 for each combination of ii, jj
            kk = (ii + n) * (2 * n + 1) + (jj + n)
            # ii and jj, left and right, start and end
            ii_l_s, ii_l_e, ii_r_s, ii_r_e = stack_slice_indices(ii)
            jj_l_s, jj_l_e, jj_r_s, jj_r_e = stack_slice_indices(jj)
            shifted_copy = floats_invalid.copy()
            shifted_copy[ii_l_s:ii_l_e, jj_l_s:jj_l_e] = y[ii_r_s:ii_r_e, jj_r_s:jj_r_e]
            local_neighborhood[kk, :] = shifted_copy.ravel()
            shifted_mask = bools_true.copy()
            #            shifted_mask[ii_l_s:ii_l_e, jj_l_s:jj_l_e] = bools_false[ii_r_s:ii_r_e, jj_r_s:jj_r_e]
            shifted_mask[ii_l_s:ii_l_e, jj_l_s:jj_l_e] = False
            neighborhood_mask[kk, :] = shifted_mask.ravel()
    return local_neighborhood, neighborhood_mask


def stack_slice_indices(ii):
    '''
    Helper function for shift_stack and square_stack to select slicing ranges.

    :param ii: int
    :return slice_indices_1, slice_indices_2: 2-element lists of ints
    '''
    if ii < 0:
        ii_left_start = None
        ii_left_end = ii
        ii_right_start = -ii
        ii_right_end = None
    elif ii == 0:
        ii_left_start = None
        ii_left_end = None
        ii_right_start = None
        ii_right_end = None
    else:  # if ii > 0
        ii_left_start = ii
        ii_left_end = None
        ii_right_start = None
        ii_right_end = -ii
    return ii_left_start, ii_left_end, ii_right_start, ii_right_end


def masked_median_2d_axis_0(y2d, mask2d):
    '''
    Takes the median of masked data along axis 0.

    :param y2d: numpy float array
    :param mask2d: numpy bool array
    :return median: 1d numpy float array
    :return mask_redux: 1d numpy bool array

    *y2d* is data; *mask2d* is its corresponding mask of the same shape
    with values *False* for legitimate data, *True* otherwise.
    Unlike mean and variance, median cannot negate elements by setting their weight to zero.
    This leads to unequally-sized regions, esp. near boundaries.
    In order to avoid unnecessary iteration over unevenly-sized boundary regions,
    those regions are split up by number of valid elements,
    e.g., all columns with 3 valid elements are handled together.
    This will most likely reduce the computation time compared to looping over each column,
    especially if *y2d* is large.
    Returns *median*, the median of *y2d* along axis 0,
    and *mask_redux*, which is *False* if a defined median exists, *True* otherwise.
    '''
    mask_redux = ~((~mask2d).any(axis=0))
    redux_shape = mask_redux.shape
    median = np.zeros(redux_shape, dtype=float)
    num_entries = (~mask2d).sum(axis=0)
    num_max_entries = num_entries.max()
    num_min_entries = num_entries.min()
    if num_min_entries == 0:
        num_min_entries = 1  # skip past empty columns; they're already masked zeros
    for i in range(num_min_entries, num_max_entries + 1):
        i_entries = np.equal(num_entries, i)
        num_i = i_entries.sum()  # number of columns with *i* entries
        if num_i != 0:  # This case for axis == 0.  Diffnt case for axis != 0?  Surely.  In fact, yeah.
            mask_i = i_entries & (~mask2d)  # i_entries is broadcast to dimensions of mask2d.  True for used elements.
            # numpy aggregates indexed items along rows first, then columns.
            # We want columns aggregated first for this application, so we get clever with transpose.
            medianees = y2d.T[mask_i.T]
            medianees = (medianees.reshape((num_i, i))).T
            median[i_entries] = np.median(medianees, axis=0)
    return median, mask_redux


def valid_local_delta(data, mask, axis, direction):
    '''
    Determines masked difference between each pixel and its neighbor in indicated direction.

    :param data: 2d numpy float array
    :param mask: 2d numpy bool array
        *True* indicates invalid data; *False* indicates valid data
    :param axis: int
        Allowed values 0, 1
    :param direction: int
        Allowed values -1, 1
    :return delta: 2d numpy float array
    :return invalid: 2d numpy bool array
        *True* indicates invalid data; *False* indicates valid data
    '''
    if data.shape != mask.shape:
        raise ValueError('Parameters "data" and "mask" must have the same shape.')
    # if data.dtype not in [int, float]:
    #        raise TypeError('Parameter "data" must contain numeric information.')
    if axis not in [0, 1]:
        raise ValueError('Parameter "axis" must be 0 or 1.')
    if direction not in [-1, 1]:
        raise ValueError('Parameter "direction" must be -1 or 1.')
    # indices for slicing
    i1, i2, i3, i4, i5, i6, i7, i8 = None, None, None, None, None, None, None, None
    if axis == 0:
        if direction == 1:
            i2, i5 = -1, 1
        elif direction == -1:
            i1, i6 = 1, -1
    elif axis == 1:
        if direction == 1:
            i4, i7 = -1, 1
        elif direction == -1:
            i3, i8 = 1, -1
    # Calculations
    delta = np.zeros(data.shape, dtype=float)
    delta[i1:i2, i3:i4] = data[i1:i2, i3:i4] - data[i5:i6, i7:i8]
    invalid = np.ones(mask.shape, dtype=bool)
    invalid[i1:i2, i3:i4] = np.logical_or(mask[i1:i2, i3:i4], mask[i5:i6, i7:i8])
    delta *= (~invalid)
    return delta, invalid


def positive_laplacian(data, full_mask):
    '''

    :param data: 2d numpy float array
    :param full_mask: 2d numpy bool array
        *True* indicates invalid data; *False* indicates valid data
    :return laplacian: 2d numpy float array
    '''
    '''
    In this function, any invalid contribution is set to zero.
    This gives the correct result for pixels near boundaries, etc.;
    they are compared to their remaining legitimate neighbors.
    '''
    # Invalid values set to zero by *valid_local_delta*
    dx_plus, _ = valid_local_delta(data, full_mask, 0, 1)
    dx_minus, _ = valid_local_delta(data, full_mask, 0, -1)
    dy_plus, _ = valid_local_delta(data, full_mask, 1, 1)
    dy_minus, _ = valid_local_delta(data, full_mask, 1, -1)
    # Add with requirement that each summed element is > 0.
    laplacian = dx_plus * (dx_plus > 0) + dx_minus * (dx_minus > 0) + dy_plus * (dy_plus > 0) \
                + dy_minus * (dy_minus > 0)
    return laplacian


def grow_four_directions(mask):
    newmask = mask.copy()
    newmask[:, 1:] = np.logical_or(newmask[:, 1:], mask[:, :-1])
    newmask[:, :-1] = np.logical_or(newmask[:, :-1], mask[:, 1:])
    newmask[1:, :] = np.logical_or(newmask[1:, :], mask[:-1, :])
    newmask[:-1, :] = np.logical_or(newmask[:-1, :], mask[1:, :])
    return newmask


def grow_eight_directions(mask):
    mask = mask.copy()
    mask[:, 1:] = np.logical_or(mask[:, 1:], mask[:, :-1])
    mask[:, :-1] = np.logical_or(mask[:, :-1], mask[:, 1:])
    mask[1:, :] = np.logical_or(mask[1:, :], mask[:-1, :])
    mask[:-1, :] = np.logical_or(mask[:-1, :], mask[1:, :])
    return mask


def square_grow(mask, n):
    mask = mask.copy()
    for i in range(n):
        mask[:, 1:] = np.logical_or(mask[:, 1:], mask[:, :-1])
        mask[:, :-1] = np.logical_or(mask[:, :-1], mask[:, 1:])
    for i in range(n):
        mask[1:, :] = np.logical_or(mask[1:, :], mask[:-1, :])
        mask[:-1, :] = np.logical_or(mask[:-1, :], mask[1:, :])
    return mask


def scipy_med_filter(y, y_out, needs_replaced, n):
    median = ndimage.filters.median_filter(y, size=(2 * n + 1), mode='mirror')
    y_out[needs_replaced] = median[needs_replaced]
    return y_out


def old_med_filter(y, y_out, needs_replaced, n, all_mask=np.zeros(1)):
    if not all_mask.any():
        all_mask = np.zeros(y.shape, dtype=bool)
    # So... mask is a 2D array containing False and True, where True means "here is a cosmic"
    cosmicindices = np.argwhere(needs_replaced)
    # This is a list of the indices of cosmic affected pixels.

    # We put cosmic ray pixels to np.Inf to flag them :
    y_out[needs_replaced] = np.Inf

    # Now we want to have a n pixel frame of Inf padding around our image.
    w, h = y_out.shape
    padarray = np.zeros((w + 2 * n, h + 2 * n)) + np.Inf
    padarray[n:w + n, n:h + n] = y_out.copy()  # that copy is important, we need 2 independent arrays
    # The medians will be evaluated in this padarray, skipping the np.Inf.
    # Now in this copy called padarray, we also put the saturated stars to np.Inf, if available :
    if all_mask.any():  #####
        padarray[n:w + n, n:h + n][all_mask] = np.Inf

    backgroundlevel = np.median(y[~all_mask])
    # A loop through every cosmic pixel :
    for cosmicpos in cosmicindices:
        ii, jj = cosmicpos
        cutout = padarray[ii:(ii + 2 * n + 1), jj:(jj + 2 * n + 1)].ravel()  # remember the shift due to the padding !
        # print cutout
        # Now we have our 25 pixels, some of them are np.Inf, and we want to take the median
        goodcutout = cutout[cutout != np.Inf]
        # print np.alen(goodcutout)

        if np.size(goodcutout) >= ((2 * n + 1) ** 2):
            # This never happened, but you never know ...
            raise RuntimeError, "Mega error in clean !"
        elif np.size(goodcutout) > 0:
            replacementvalue = np.median(goodcutout)
        else:
            # i.e. no good pixels : Shit, a huge cosmic, we will have to improvise ...
            print '''VERY large, blocky cosmic detected.  Check results for error.'''
            replacementvalue = backgroundlevel
        # We update the cleanarray,
        # but measure the medians in the padarray, so to not mix things up...
        y_out[ii, jj] = replacementvalue

    # That's it.
    print "Cleaning done"
    return y_out


def alt_old_med_filter(y, y_out, needs_replaced, n, all_mask=np.zeros(1)):
    if not all_mask.any():
        all_mask = np.zeros(y.shape, dtype=bool)
    # So... mask is a 2D array containing False and True, where True means "here is a cosmic"
    cosmicindices = np.argwhere(needs_replaced)
    # This is a list of the indices of cosmic affected pixels.

    good_pixels = (~all_mask) & (~needs_replaced)
    height, width = y.shape
    index0 = np.reshape(np.arange(height), (height, 1)) * np.ones((height, width))
    index1 = np.reshape(np.arange(width), (1, width)) * np.ones((height, width))

    # We put cosmic ray pixels to np.Inf to flag them :
    #    y_out[needs_replaced] = np.Inf
    # Now we want to have a 2 pixel frame of Inf padding around our image.
    #    w = y_out.shape[0]
    #    h = y_out.shape[1]
    #    padarray = np.zeros((w + 4, h + 4)) + np.Inf
    #    padarray[2:w + 2, 2:h + 2] = y_out.copy()  # that copy is important, we need 2 independent arrays
    # The medians will be evaluated in this padarray, skipping the np.Inf.
    # Now in this copy called padarray, we also put the saturated stars to np.Inf, if available :
    #    if all_mask.any():
    #        padarray[2:w + 2, 2:h + 2][all_mask] = np.Inf

    backgroundlevel = np.median(y[~all_mask])
    # A loop through every cosmic pixel :
    for cosmic_position in cosmicindices:
        jj, ii = cosmic_position
        jj1 = max(jj - n, 0)
        jj2 = min(jj + n, height - 1)
        ii1 = max(ii - n, 0)
        ii2 = min(ii + n, width - 1)
        neighbors = (index0 >= jj - n) & (index0 <= jj + n) & (index1 >= ii - n) & (index1 <= ii + n)
        good_neighbors = neighbors & good_pixels
        good_cutout = y[good_neighbors]
        #        ii_pad = cosmicpos[0] + 2 - n
        #        jj_pad = cosmicpos[1] + 2 - n
        #        ii_out = cosmicpos[0]
        #        jj_out = cosmicpos[1]
        #        cutout = padarray[ii_pad:ii_pad + 2*n+1, jj_pad:jj_pad + 2*n+1].ravel()  # remember the shift due to the padding !
        # print cutout
        # Now we have our 25 pixels, some of them are np.Inf, and we want to take the median
        #        goodcutout = cutout[cutout != np.Inf]
        # print np.alen(goodcutout)

        if np.size(good_cutout) >= ((2 * n + 1) ** 2):
            # This never happened, but you never know ...
            raise RuntimeError, "Mega error in clean !"
        elif np.size(good_cutout) > 0:
            replacementvalue = np.median(good_cutout)
        else:
            # i.e. no good pixels : Shit, a huge cosmic, we will have to improvise ...
            print "OH NO, I HAVE A HUUUUUUUGE COSMIC !!!!!"
            replacementvalue = backgroundlevel
        # We update the cleanarray,
        # but measure the medians in the padarray, so to not mix things up...
        y_out[jj, ii] = replacementvalue

    # That's it.
    print "Cleaning done"
    return y_out


def new_med_filter(y, y_out, needs_replaced, n, all_mask=np.zeros(1)):
    if not all_mask.any():
        all_mask = np.zeros(y.shape, dtype=bool)
    neighborhood, shift_mask = square_stack(y, n)
    all_mask_neighborhood, _ = square_stack(all_mask, n)  # Do not use values from pixels masked for any reason
    all_mask_shift_mask = all_mask_neighborhood * shift_mask
    median, median_mask = masked_median_2d_axis_0(neighborhood, all_mask_shift_mask)
    median = median.reshape(y.shape)
    median_mask = median_mask.reshape(y.shape)  # True = undefined, False = defined

    replaceable_mask = ((~median_mask) & needs_replaced)  # True = can be replaced with masked median
    irreplaceable_mask = (median_mask & needs_replaced)  # True = cannot be replaced with masked median

    y_out[replaceable_mask] = median[replaceable_mask]
    # Next clause triggers only if there are 5x5 or larger zingers or holes.
    if irreplaceable_mask.any():
        backgroundlevel = np.median(y[~all_mask])
        y_out[irreplaceable_mask] = backgroundlevel
    return y_out


def gen_test_arrays(l, p):
    d = np.random.rand(l, l)
    dout = d.copy()
    dmask = (np.random.rand(l, l) < p)
    return d, dout, dmask


# d, dout, dmask = gen_test_arrays(1024, 0.1)

'''
def subsample(a):  # this is more a generic function then a method ...
    """
    Returns a 2x2-subsampled version of array a (no interpolation, just cutting pixels in 4).
    The version below is directly from the scipy cookbook on rebinning :
    U{http://www.scipy.org/Cookbook/Rebinning}
    There is ndimage.zoom(cutout.array, 2, order=0, prefilter=False), but it makes funny borders.

    """
    newshape = (2 * a.shape[0], 2 * a.shape[1])
    slices = [slice(0, old, float(old) / new) for old, new in zip(a.shape, newshape)]
    coordinates = np.mgrid[slices]
    indices = coordinates.astype('i')  #choose the biggest smaller integer index
    return a[tuple(indices)]


def rebin(a, newshape):
    """
    Auxiliary function to rebin an ndarray a.
    U{http://www.scipy.org/Cookbook/Rebinning}

            >>> a=rand(6,4); b=rebin(a,(3,2))
    """

    shape = a.shape
    lenShape = len(shape)
    factor = np.asarray(shape) / np.asarray(newshape)
    #print factor
    evList = ['a.reshape('] + \
             ['newshape[%d],factor[%d],' % (i, i) for i in xrange(lenShape)] + \
             [')'] + ['.sum(%d)' % (i + 1) for i in xrange(lenShape)] + \
             ['/factor[%d]' % i for i in xrange(lenShape)]

    return eval(''.join(evList))


def rebin2x2(a):
    """
    Wrapper around rebin that actually rebins 2 by 2
    """
    inshape = np.array(a.shape)
    if not (inshape % 2 == np.zeros(2)).all():  # Modulo check to see if size is even
        raise RuntimeError, "I want even image shapes !"

    return rebin(a, inshape / 2)
'''
