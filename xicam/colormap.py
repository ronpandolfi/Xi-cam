import numpy as np


Gray = np.arange(255)

Red = np.round(255.0 * (np.sin((2.0 * Gray * np.pi / 255.0)) + 1.0) / 2.0)

Green = np.round(255.0 * (np.sin((2.0 * Gray * np.pi / 255.0) - (np.pi / 2.0)) + 1.0) / 2.0)

Blue = np.round(255.0 * (np.sin((2.0 * Gray * np.pi / 255.0) - (np.pi)) + 1.0) / 2.0)

LUT = np.array([Red, Green, Blue]).T

# print LUT
#print LUT.shape