"""
About
=====

cosmics.py is a small and simple python module to detect and clean cosmic ray hits on images (numpy arrays or FITS), using scipy, and based on Pieter van Dokkum's L.A.Cosmic algorithm.

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
	This feature is mainly for pretty-image production. It is optional, requires one more parameter
	(a CCD saturation level in ADU), and uses some nicely robust morphology operations and object extraction.

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

# We define the laplacian kernel to be used
laplkernel = np.array([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]])


class Cosmics(Operation):
    def __init__(self):
        pass

    def secondInit(self, rawarray, gain=1.0, readnoise=0.0, satlevel=65536, pssl=0.0, sigclip=5.0, sigfrac=0.3,
                 objlim=5.0, premask=np.zeros(1), verbose=True):
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
        ##gain0  string statsec = "*,*"       # section to use for automatic computation of gain
        real   skyval  = 0.           # sky level that has been subtracted (ADU)
        real   sigclip = 3.0          # detection limit for cosmic rays (sigma)
        real   sigfrac = 0.5          # fractional detection limit for neighbouring pixels
        real   objlim  = 3.0           # contrast limit between CR and underlying object
        int    niter   = 1            # maximum number of iterations

        """
        self.rawarray = rawarray + pssl + 0.0  # internally, we will always work "with sky".  Also forces cast to floats.
        self.cleanarray = self.rawarray.copy()  # In lacosmiciteration() we work on this guy
        self.mask = np.cast['bool'](np.zeros(self.rawarray.shape))  # All False, no cosmics yet
        if premask.any():  # Optional pre-masked region
            self.premask = premask
        else:
            self.premask = np.zeros(self.rawarray.shape, dtype=bool)

        self.gain = gain
        self.readnoise = readnoise
        self.sigclip = sigclip
        self.objlim = objlim
        self.sigcliplow = sigclip * sigfrac
        self.satlevel = satlevel

        self.verbose = verbose

        self.pssl = pssl

        self.backgroundlevel = None  # only calculated and used if required.
        self.satstars = np.zeros(1, dtype=float)  # a mask of the saturated stars, only calculated if required
        self.med3 = np.zeros(1, dtype=float)
        self.med5 = np.zeros(1, dtype=float)
        self.med37 = np.zeros(1, dtype=float)

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

        if self.satstars.any():
            stringlist.append("Saturated star mask : %i pixels" % np.sum(self.satstars))

        return "\n".join(stringlist)

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
        dilmask = grow_four_directions(square_grow(self.mask, 1))
        # origin = 0 means center
        (labels, n) = ndimage.measurements.label(dilmask)
        slicecouplelist = ndimage.measurements.find_objects(labels)
        # Now we have a huge list of couples of numpy slice objects giving a frame around each object
        # For plotting purposes, we want to transform this into the center of each object.
        centers = [[(tup[0].start + tup[0].stop) / 2.0, (tup[1].start + tup[1].stop) / 2.0] for tup in slicecouplelist]
        # We also want to know how many pixels were affected by each cosmic ray.
        # Why ? Dunno... it's fun and available in scipy :-)
        sizes = ndimage.measurements.sum(self.mask.ravel(), labels.ravel(), np.arange(1, n + 1, 1))
        retdictlist = [{"name": "%i" % size, "x": center[0], "y": center[1]} for (size, center) in zip(sizes, centers)]

        if verbose:
            print "Labeling done"

        return retdictlist

    def getdilatedmask(self, size=3):
        """
        Returns a morphologically dilated copy of the current mask.
        size = 3 or 5 decides how to dilate.
        """
        if size == 3:
            dilmask = square_grow(self.mask, 1)
        elif size == 5:
            dilmask = grow_four_directions(square_grow(self.mask, 1))
        else:
            # dilmask = self.mask.copy()
            raise ValueError("Argument *size* of *getdilatedmask* should be either 3 or 5.")
        return dilmask

    def clean(self, mask=None, verbose=None):
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
            mask = self.mask

        if verbose:
            print "Cleaning cosmic affected pixels ..."

        self.update_median_filters()
        self.cleanarray[mask] = self.med5[mask]
        # Generally, cleaning is done at this point.

        # This section will only trigger if there are masked regions larger than 5x5.
        infinite_entries = (np.isinf(self.cleanarray) & ~self.premask)
        median_size = 5
        while infinite_entries.any() and (median_size <= 9):
            # Tell user
            num_huge_cosmics = infinite_entries.sum()
            print '%i pixel(s) were in the middle of huuuuuuuuge cosmics.' % num_huge_cosmics
            # Employ alternative
            median_size += 2
            ignore_mask = self.mask & self.premask
            if self.satstars.any():
                ignore_mask = ignore_mask & self.satstars
            bigger_median = targeted_masked_median(self.rawarray, infinite_entries, median_size, ignore_mask)
            self.cleanarray[infinite_entries] = bigger_median[infinite_entries]
            # See if any are left.
            infinite_entries = (np.isinf(self.cleanarray) & ~self.premask)

        if infinite_entries.any():  # If all else fails (masked regions larger than 9x9)
            backgroundlevel = self.guessbackgroundlevel()
            self.cleanarray[infinite_entries] = backgroundlevel

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
        # The subtlety is that we want to include all saturated pixels connected to these saturated stars...
        # I haven't found a better solution then the double loop

        # We dilate the satpixels alone, to ensure connectivity in glitchy regions and to add a safety margin around them.
        dilsatpixels = grow_four_directions(square_grow(satpixels, 1))
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

    def getsatstars(self, verbose=None):
        """
        Returns the mask of saturated stars after finding them if not yet done.
        Intended mainly for external use.
        """
        if verbose == None:
            verbose = self.verbose
        if not self.satlevel > 0:
            raise RuntimeError, "Cannot determine satstars : you gave satlevel <= 0 !"
        if not self.satstars.any():
            self.findsatstars(verbose=verbose)
        return self.satstars

    def getmask(self):
        return self.mask

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

    def guessbackgroundlevel(self):
        """
        Estimates the background level. This could be used to fill pixels in large cosmics.
        """
        if self.backgroundlevel == None:
            # self.backgroundlevel = np.median(self.rawarray) # 0.071662902832 s
            self.backgroundlevel = np.median(self.rawarray.ravel())  # 0.0666739940643 s
        return self.backgroundlevel

    ### whhhhaaa
    def positive_laplacian2(self):
        laplacian = np.zeros(self.rawarray.shape)
        l_0_plus = self.rawarray[1:, :] - self.rawarray[:-1, :]
        l_0_minus = -l_0_plus  # y[:-1, :] - y[1:, :]
        l_1_plus = self.rawarray[:, 1:] - self.rawarray[:, :-1]
        l_1_minus = -l_1_plus  # y[:, :-1] - y[:, 1:]
        laplacian[1:, :] += l_0_plus.clip(min=0)
        laplacian[:-1, :] += l_0_minus.clip(min=0)
        laplacian[:, 1:] += l_1_plus.clip(min=0)
        laplacian[:, :-1] += l_1_minus.clip(min=0)
        return laplacian

    def positive_laplacian(self):
        # We subsample, convolve, clip negative values, and rebin to original size
        subsampled = subsample(self.rawarray)
        convolved = signal.convolve2d(subsampled, laplkernel, mode="same", boundary="symm")
        clipped = convolved.clip(min=0.0)
        lplus = rebin2x2(clipped)
        return lplus

    def positive_laplacian3(self):
        mask = self.mask | self.premask
        if self.satstars.any():
            mask = mask | self.satstars
        l = np.zeros(self.rawarray.shape, dtype=float)
        lmask = np.zeros(self.rawarray.shape, dtype=bool)
        # nmask = np.zeros(self.rawarray.shape, dtype=int)
        l_0_plus = self.rawarray[1:, :] - self.rawarray[:-1, :]
        l_0_plus_mask = mask[1:, :] & mask[:-1, :]
        l_0_minus = -l_0_plus  # y[:-1, :] - y[1:, :]
        l_0_minus_mask = l_0_plus_mask  # mask[:-1, :] & mask[1:, :]
        l_1_plus = self.rawarray[:, 1:] - self.rawarray[:, :-1]
        l_1_plus_mask = mask[:, 1:] & mask[:, :-1]
        l_1_minus = -l_1_plus  # y[:, :-1] - y[:, 1:]
        l_1_minus_mask = l_1_plus_mask  # mask[:, :-1] & mask[:, 1:]
        l[1:, :] += l_0_plus.clip(min=0) * ~l_0_plus_mask
        l[:-1, :] += l_0_minus.clip(min=0) * ~l_0_minus_mask
        l[:, 1:] += l_1_plus.clip(min=0) * ~l_1_plus_mask
        l[:, :-1] += l_1_minus.clip(min=0) * ~l_1_minus_mask
        lmask[1:, :] = (lmask[1:, :] | l_0_plus_mask)
        lmask[:-1, :] = (lmask[:-1, :] | l_0_minus_mask)
        lmask[:, 1:] = (lmask[:, 1:] | l_1_plus_mask)
        lmask[:, :-1] = (lmask[:, :-1] | l_1_plus_mask)
        return l  # , lmask

    def noise_model(self):
        # We build a custom noise map, so to compare the laplacian to
        # m5 = ndimage.filters.median_filter(self.cleanarray, size=5, mode='mirror')
        m5 = self.med5
        # We keep this m5, as I will use it later for the interpolation.
        m5clipped = m5.clip(min=0.0)  # As we will take the sqrt
        noise = (1.0 / self.gain) * np.sqrt(self.gain * m5clipped + self.readnoise * self.readnoise)
        return noise

    def create_median_filters(self):
        self.med3 = hybrid_masked_median(self.rawarray, 3, self.premask)
        self.med5 = hybrid_masked_median(self.rawarray, 5, self.premask)
        self.med37 = hybrid_masked_median(self.med3, 7, self.premask)

    def update_median_filters(self):
        ignore_mask = self.mask & self.premask
        if self.satstars.any():
            ignore_mask = ignore_mask & self.satstars
        fix_mask = (self.mask & (~self.premask))
        if self.satstars.any():
            fix_mask = (fix_mask & (~self.satstars))
        med3update = targeted_masked_median(self.rawarray, fix_mask, 3, ignore_mask)
        self.med3[fix_mask] = med3update[fix_mask]
        med5update = targeted_masked_median(self.rawarray, fix_mask, 5, ignore_mask)
        self.med5[fix_mask] = med5update[fix_mask]
        med37mask = np.isinf(self.med3) | self.premask  ### ?????
        self.med37 = hybrid_masked_median(self.med3, 7, med37mask)
        # Re-calculate areas near newly masked pixels
        # but not pre-masked areas

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
            print "Updating masked median filters..."
        if not self.med5.any():
            self.create_median_filters()
        else:
            self.update_median_filters()

        if verbose:
            print "Convolving image with Laplacian kernel ..."
        lplus = self.positive_laplacian()

        if verbose:
            print "Creating noise model ..."
        noise = self.noise_model()

        if verbose:
            print "Calculating Laplacian signal to noise ratio ..."
        sigmap = lplus / (2.0 * noise)  # the 2.0 is from the 2x2 subsampling
        # We remove the large structures (s prime) :
        s_prime = sigmap - hybrid_masked_median(sigmap, 5)  ## mask??

        if verbose:
            print "Selecting sharp-edged features as candidate cosmic rays ..."
        # Candidate cosmic rays (this will include stars + HII regions)
        candidates = s_prime > self.sigclip
        nbcandidates = np.sum(candidates)
        if verbose:
            print "  %5i candidate pixels" % nbcandidates

        # At this stage we use the saturated stars to mask the candidates, if available :
        if self.satstars.any():
            if verbose:
                print "Masking saturated stars ..."
            candidates = ((~self.satstars) & candidates)
            nbcandidates = np.sum(candidates)
            if verbose:
                print "  %5i candidate pixels not part of saturated stars" % nbcandidates

        if verbose:
            print "Building fine structure image ..."
        # We build the fine structure image :
        fine_structure = self.med3 - self.med37
        # In the article that's it, but in lacosmic.cl fine_structure is divided by the noise...
        # Ok I understand why, it depends on if you use s_prime/fine_structure or lplus/fine_structure as criterion.
        # s_prime, unlike lplus, has already been divided by noise once.
        # There are some differences between the article and the iraf implementation.
        # So I will stick to the iraf implementation.
        fine_structure = fine_structure / noise
        fine_structure = fine_structure.clip(min=0.0)  # as we will divide by f. like in the iraf version.

        if verbose:
            print "Removing suspected compact bright objects ..."
        # Now we have our better selection of cosmics :
        cosmics = (candidates & ((s_prime / fine_structure) > self.objlim))
        # Note the s_prime/fine_structure and not lplus/fine_structure ...
        # ... due to the fine_structure = fine_structure/noise above.

        nbcosmics = np.sum(cosmics)
        if verbose:
            print "  %5i remaining candidate pixels" % nbcosmics

        # What follows is a special treatment for neighbors of detected cosmics.
        # They are treated as more likely to be cosmics than the general population.
        if verbose:
            print "Finding neighboring pixels affected by cosmic rays ..."

        # We grow these cosmics a first time to determine the immediate neighborhod  :
        growcosmics = square_grow(cosmics, 1)
        # From this grown set, we keep those that have s_prime > sigmalim.
        # Note that the newly selected pixels did NOT meet both s_prime > self.sigclip and
        # (s_prime / fine_structure) > self.objlim), or they'd have been selected already.
        # Of these candidates, we require only the first condition.
        growcosmics = (s_prime > self.sigclip) & growcosmics
        # We repeat this procedure, but lower the detection limit to sigmalimlow :
        finalsel = square_grow(growcosmics, 1)
        finalsel = ((s_prime > self.sigcliplow) & finalsel)

        # Again, we have to kick out pixels on saturated stars :
        if self.satstars.any():
            if verbose:
                print "Masking saturated stars ..."
            finalsel = ((~self.satstars) & finalsel)

        nbfinal = np.sum(finalsel)
        if verbose:
            print "  %5i pixels detected as cosmics" % nbfinal

        # We find how many cosmics are not previously known :
        newmask = ((~self.mask) & finalsel)
        nbnew = np.sum(newmask)

        # We update the mask with the cosmics we have found :
        self.mask = (self.mask | finalsel)

        # We return
        # (used by function lacosmic)

        return {"niter": nbfinal, "nnew": nbnew, "itermask": finalsel, "newmask": newmask}

    def negative_laplacian(self):
        # We subsample, convolve, clip negative values, and rebin to original size
        subsampled = subsample(self.cleanarray)
        convolved = -signal.convolve2d(subsampled, laplkernel, mode="same", boundary="symm")
        clipped = convolved.clip(min=0.0)
        lplus = rebin2x2(clipped)
        return lplus

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

        # The holes are the peaks in this image that are not stars

        #holes = h > 300
        """
        """
        if verbose == None:
            verbose = self.verbose

        if verbose :
            print "Finding holes ..."

        lplus = self.negative_laplacian()

        if not self.med5.any():
            self.create_median_filters()
        else:
            self.update_median_filters()
        noise = self.noise_model()

        sigmap = lplus / (2.0 * noise) # the 2.0 is from the 2x2 subsampling

        # We remove the large structures (s prime) :
        s_prime = sigmap - hybrid_masked_median(sigmap, 5) ## mask??

        holes = s_prime > self.sigclip
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

        if self.satlevel > 0 and not self.satstars.any():
            self.findsatstars(verbose=True)

        print "Starting %i L.A.Cosmic iterations ..." % maxiter
        ii = 1
        nnew = np.nan
        while (ii <= maxiter) & (nnew != 0):
            print "Iteration %i" % ii
            iterres = self.lacosmiciteration(verbose=verbose)
            # iterres = {"niter": nbfinal,       # Number cosmics detected by this iteration
            #           "nnew": nbnew,          # Number cosmics not previously known
            #           "itermask": finalsel,   # Cosmics detected by this iteration
            #           "newmask": newmask}     # All known cosmics

            print "%i cosmic pixels (%i new)" % (iterres["niter"], iterres["nnew"])
            if iterres["nnew"] == 0:
                print "All detectable zingers have been identified."
            elif ii == maxiter:
                print "Maximum iterations preformed.  More zingers may remain."

            nnew = iterres["nnew"]
            ii += 1

        self.clean(verbose=verbose)
        # Note that for huge cosmics, one might want to revise this.
        # Thats why I added a feature to skip saturated stars !


# Array manipulation

def subsample(a):  # this is more a generic function than a method ...
    """
    Returns a 2x2-subsampled version of array a (no interpolation, just cutting pixels in 4).
    """
    big_a = np.zeros((a.shape[0] * 2, a.shape[1] * 2), a.dtype)
    big_a[::2, ::2] = a
    big_a[1::2, ::2] = a
    big_a[::2, 1::2] = a
    big_a[1::2, 1::2] = a
    return big_a


def rebin2x2(a):
    """
    Returns the average value of 2pix by 2pix regions.
    """
    h, w = a.shape
    if ((h / 2) * 2 != h) or ((w / 2) * 2 != w):
        raise ValueError("The input *a* to function *rebin2x2* must have an even number of columns and rows.")
    # new_a = np.zeros((h/2, w/2), dtype=a.dtype)
    new_a = (a[::2, ::2] + a[1::2, ::2] + a[::2, 1::2] + a[1::2, 1::2]) / 4.0
    return new_a


def targeted_masked_median(y, target_mask, n, legitimacy_mask):
    # The medians will be evaluated from this padarray, skipping the np.inf.
    padarray = inf_padded_array(y, n)
    padarray[n:-n, n:-n][legitimacy_mask] = np.inf
    # Prep recipient array
    med = np.zeros(y.shape)  # 0.011280 seconds
    target_indices = np.argwhere(target_mask)
    for location in target_indices:
        ii, jj = location
        cutout = padarray[ii:(ii + 2 * n + 1), jj:(jj + 2 * n + 1)].ravel()  # remember the shift due to the padding !
        # Of our (2*n+1)**2 pixels, some of them are masked/np.inf and should be excluded
        goodcutout = cutout[cutout != np.inf]

        if np.size(goodcutout) > 0:
            replacementvalue = np.median(goodcutout)
        else:
            # i.e. no good pixels
            replacementvalue = np.inf
        med[ii, jj] = replacementvalue
    return med


def hybrid_masked_median(y, n, ignore_mask=np.zeros(1)):
    '''Combines maskless scipy median with true masked median for select areas.

    :param y:
    :param n:
    :param ignore_mask:
    :return:

    As the scipy ndimage.filters.median_filter routine is quite fast,
    it is used to initially calculate the median.  The median is then corrected
    to the masked-median value for edge regions and regions near masked pixels.

    This produces identical results to the full masked median,
    but in some cases this may represent a significant computation savings
    over directly applying the masked median to the full image.
    '''
    if (int(n) != ((int(n) / 2) * 2 + 1)) or (n < 0):
        raise ValueError("Argument *n* of *hybrid_masked_median* should be a positive odd whole number.")
    m = int(n) / 2  # 2*m + 1 = n

    # Make simple median
    med = ndimage.filters.median_filter(y, size=n, mode='mirror')

    # Re-calculate areas with edge effects and areas near pre-masked pixels
    edge_mask = edge_mask_array(y, m)
    if ignore_mask.any():
        pre_mask = square_grow(ignore_mask, m)
        fixit_mask = ((edge_mask | pre_mask) & ~ignore_mask)
    else:
        fixit_mask = edge_mask
    masked_median = targeted_masked_median(y, fixit_mask, n, fixit_mask)
    med[fixit_mask] = masked_median[fixit_mask]
    return med


def inf_padded_array(y, n):
    # Now we want to have a n pixel frame of Inf padding around our image.
    w, h = y.shape
    # padarray = np.zeros((w + 2*n, h + 2*n)) + np.inf
    padarray = np.zeros((w + 2 * n, h + 2 * n))
    padarray = padarray + np.inf
    padarray[n:-n, n:-n] = y.copy()  # copy to ensure no overwrite
    return padarray


def square_grow(mask, m):
    mask = mask.copy()
    for i in range(m):
        mask[:, 1:] = (mask[:, 1:] | mask[:, :-1])
        mask[:, :-1] = (mask[:, :-1] | mask[:, 1:])
    for i in range(m):
        mask[1:, :] = (mask[1:, :] | mask[:-1, :])
        mask[:-1, :] = (mask[:-1, :] | mask[1:, :])
    return mask


def grow_four_directions(mask):
    newmask = mask.copy()
    newmask[:, 1:] = (newmask[:, 1:] | mask[:, :-1])
    newmask[:, :-1] = (newmask[:, :-1] | mask[:, 1:])
    newmask[1:, :] = (newmask[1:, :] | mask[:-1, :])
    newmask[:-1, :] = (newmask[:-1, :] | mask[1:, :])
    return newmask


def edge_mask_array(y, m):
    edge_mask = np.zeros(y.shape, dtype=bool)
    edge_mask[:m, :] = True
    edge_mask[-m:, :] = True
    edge_mask[:, :m] = True
    edge_mask[:, -m:] = True
    return edge_mask


'''

t0 = time.time()
#b = rebin2x2_a(a) # 0.233438 seconds
#b = rebin2x2_b(a) # 0.137624 seconds
b = rebin2x2_c(a) # 0.087244 seconds
t1 = time.time()
print "%f seconds" % (t1 - t0)
'''

'''
def positive_laplacian1(y):
    # We subsample, convolve, clip negative values, and rebin to original size
    subsampled = subsample(y)
    convolved = signal.convolve2d(subsampled, laplkernel, mode="same", boundary="symm")
    clipped = convolved.clip(min=0.0)
    lplus = rebin2x2(clipped)
    return lplus

def positive_laplacian2(y):
    l = np.zeros(y.shape)
    l_0_plus = y[1:,:] - y[:-1,:]
    l_0_minus = -l_0_plus               # y[:-1, :] - y[1:, :]
    l_1_plus = y[:, 1:] - y[:, :-1]
    l_1_minus = -l_1_plus               # y[:, :-1] - y[:, 1:]
    l[1:,:] += l_0_plus.clip(min=0)
    l[:-1,:] += l_0_minus.clip(min=0)
    l[:,1:] += l_1_plus.clip(min=0)
    l[:,:-1] += l_1_minus.clip(min=0)
    return l

def positive_laplacian3(y, mask):
    l = np.zeros(y.shape)
    lmask = np.zeros(y.shape, dtype=bool)
    #nmask = np.zeros(y.shape, dtype=int)
    l_0_plus = y[1:,:] - y[:-1,:]
    l_0_plus_mask = mask[1:,:] & mask[:-1,:]
    l_0_minus = -l_0_plus               # y[:-1, :] - y[1:, :]
    l_0_minus_mask = l_0_plus_mask      # mask[:-1, :] & mask[1:, :]
    l_1_plus = y[:, 1:] - y[:, :-1]
    l_1_plus_mask = mask[:, 1:] & mask[:, :-1]
    l_1_minus = -l_1_plus               # y[:, :-1] - y[:, 1:]
    l_1_minus_mask = l_1_plus_mask      # mask[:, :-1] & mask[:, 1:]
    l[1:,:] += l_0_plus.clip(min=0) * ~l_0_plus_mask
    l[:-1,:] += l_0_minus.clip(min=0) * ~l_0_minus_mask
    l[:,1:] += l_1_plus.clip(min=0) * ~l_1_plus_mask
    l[:,:-1] += l_1_minus.clip(min=0) * ~l_1_minus_mask
    lmask[1:, :] = (lmask[1:, :] | l_0_plus_mask)
    lmask[:-1, :] = (lmask[:-1, :] | l_0_minus_mask)
    lmask[:, 1:] = (lmask[:, 1:] | l_1_plus_mask)
    lmask[:, :-1] = (lmask[:, :-1] | l_1_plus_mask)
    return l, lmask
'''




# med = a.copy() * 0.0 # 0.054373 seconds
# med = np.zeros(a.shape) # 0.011280 seconds

# growkernel = np.ones((3, 3))
# growcosmics = np.cast['bool'](signal.convolve2d(np.cast['float32'](a), growkernel, mode="same",
#                                                boundary="symm")) # 0.354112 seconds
# growcosmics = ndimage.morphology.binary_dilation(a, structure=growkernel, iterations=1, mask=None, output=None,
#                                                 border_value=0, origin=0, brute_force=False) # 0.177301 seconds
# growcosmics = square_grow(a, 1) # 0.009588 seconds

# dilstruct = np.ones((5, 5))  # dilation structure for some morphological operations
# dilstruct[ 0,  0] = 0
# dilstruct[ 0, -1] = 0
# dilstruct[-1,  0] = 0
# dilstruct[-1, -1] = 0
# So this dilstruct looks like :
# 01110
# 11111
# 11111
# 11111
# 01110
# and is used to dilate saturated stars and connect cosmic rays.
# dilmask = ndimage.morphology.binary_dilation(a, structure=dilstruct, iterations=1, mask=None, output=None,
#                                             border_value=0, origin=0, brute_force=False) # 0.337534 seconds
# dilmask = grow_four_directions(square_grow(a, 1)) # 0.024360 seconds

# big_a = subsample1(a) # 1.02948403358 seconds for old version
# big_a = subsample2(a) # 0.111756086349 seconds, new version

# b = rebin2x2_a(a) # 0.233438 seconds old version
# b = rebin2x2_b(a) # 0.137624 seconds via slicing tricks
# b = rebin2x2_c(a) # 0.087244 seconds removed unnecessary line



###  The laplacians were... causing trouble.  I may know why.
# l1 = positive_laplacian1(a) # 1.681841 seconds
# l2 = positive_laplacian2(a, amask) # 0.168197 seconds
# l3, l3mask = positive_laplacian3(a, amask) # 0.207101 seconds pre-clip, 0.273437 seconds post-clip, 0.384312 seconds post-mask
