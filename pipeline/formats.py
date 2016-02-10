import fabio, pyFAI
from fabio.fabioimage import fabioimage
from fabio import fabioutils
from pyFAI import detectors
import numpy as np
import sys, logging

logger = logging.getLogger("openimage")


class rawimage(fabioimage):
    def read(self, f, frame=None):
        with open(f, 'r') as f:
            data = np.fromfile(f, dtype=np.int32)
        for name, detector in detectors.ALL_DETECTORS.iteritems():
            if hasattr(detector, 'MAX_SHAPE'):
                # print name, detector.MAX_SHAPE, imgdata.shape[::-1]
                if np.prod(detector.MAX_SHAPE) == len(data):  #
                    detector = detector()
                    print 'Detector found: ' + name
                    break
            if hasattr(detector, 'BINNED_PIXEL_SIZE'):
                # print detector.BINNED_PIXEL_SIZE.keys()
                if len(data) in [np.prod(np.array(detector.MAX_SHAPE) / b) for b in
                                 detector.BINNED_PIXEL_SIZE.keys()]:
                    detector = detector()
                    print 'Detector found with binning: ' + name
                    break
        data.shape = detector.MAX_SHAPE
        self.data = data
        return self


fabioutils.FILETYPES['raw'] = ['raw']
from fabio import mar345image

# OVERRIDE FABIO LOADER! fabio's loader doesn't support exterior image classes, so this will inject support for now :(
def _openimage(filename):
    """
    determine which format for a filename
    and return appropriate class which can be used for opening the image
    """
    lower_filename = filename.lower()
    for prefix in fabio.openimage.URL_PREFIX:
        if lower_filename.startswith(prefix):
            filename = filename[len(prefix):]
            if filename.startswith("//"):
                filename = filename[2:]
            if fabio.openimage.URL_PREFIX[prefix]:  # process :path[slice,:,:]
                if "[" in filename:
                    filename = filename[:filename.index("[")]
                if ":" in filename:
                    col_split = filename.split(":")
                    filename = ":".join(col_split[:-1])

    try:
        imo = fabioimage()
        byts = imo._open(filename).read(18)
        filetype = fabio.openimage.do_magic(byts)
        if filetype == "marccd" and filename.find("mccd") == -1:
            # Cannot see a way around this. Need to find something
            # to distinguish mccd from regular tif...
            filetype = "tif"
    except IOError as error:
        logger.error("%s: File probably does not exist", error)
        raise error
    except:
        try:
            file_obj = fabio.openimage.FilenameObject(filename=filename)
            if file_obj == None:
                raise Exception("Unable to deconstruct filename")
            if (file_obj.format is not None) and \
                            len(file_obj.format) != 1 and \
                            type(file_obj.format) != type(["list"]):
                # one of OXD/ ADSC - should have got in previous
                raise Exception("openimage failed on magic bytes & name guess")
            filetype = file_obj.format
            # UNUSED filenumber = file_obj.num
        except Exception as error:
            logger.error(error)
            import traceback

            traceback.print_exc()
            raise Exception("Fabio could not identify " + filename)
    klass_name = "".join(filetype) + 'image'
    module = sys.modules.get("fabio." + klass_name, None)
    if module is not None:
        if hasattr(module, klass_name):
            klass = getattr(module, klass_name)
        else:
            raise Exception("Module %s has no image class" % module)
    else:
        try:  # These lines added by RP
            klass = globals()[klass_name]  #
        except:  #
            raise Exception("Filetype not known %s %s" % (filename, klass_name))
    obj = klass()
    # skip the read for read header
    return obj


fabio.openimage._openimage = _openimage