from scipy.signal import savgol_filter
import numpy as np
import matplotlib.pyplot as plt


import pyFAI
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import signaltonoise

# open MARCCD tiff image
path = 'C:\\Research_FangRen\\Publications\\on_the_fly_paper\\Sample_data\\'
im = Image.open(path + 'LaB6.tif')
# change image object into an array
imArray = np.array(im)
s = int(imArray.shape[0])
im.close()

detector_mask = np.ones((s,s))*(imArray <= 0)


# parameters I originally used.
d_in_pixel = 2462.69726489     # distance from sample to detector plane along beam direction in pixel space
Rot = (np.pi*2-4.69729438873)/(2*np.pi)*360  #detector rotation
tilt = 0.503226642865/(2*np.pi)*360   # detector tilt
lamda = 0.97621599151  # wavelength
x0 = 969.878684978     # beam center in pixel-space
y0 = 2237.93277884    # beam center in pixel-space
PP = 0.95   # beam polarization, decided by beamline setup


pixelsize = 79    # measured in microns
d = d_in_pixel*pixelsize*0.001  # measured in milimeters

p = pyFAI.AzimuthalIntegrator(wavelength=lamda)
p.setFit2D(d,x0,y0,tilt,Rot,pixelsize,pixelsize)

Qlist, IntAve = p.integrate1d(imArray, 1000, mask = detector_mask, polarization_factor = PP)
Qlist = Qlist * 10e8

spectrum = np.concatenate(([Qlist], [IntAve]))

np.savetxt(path + 'LaB6.csv', spectrum.T, delimiter= ',')