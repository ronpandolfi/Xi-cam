"""
author: fangren
"""

import numpy as np

def extract_metadata(filepath):
    """
    extract metadata from the txt files associated with the images, including the coordinates,
    """
    txtpath = filepath[:-3]+'txt'
    txt = open(txtpath)
    data = txt.readlines()
    txt.close()

    # parsing the counters
    counters = data[7].split(', ')
    exposure_time = float([counter[4:] for counter in counters if counter.startswith('sec=')][0])
    I0 = float([counter[3:] for counter in counters if counter.startswith('i0=')][0])
    I1 = float([counter[3:] for counter in counters if counter.startswith('i1=')][0])
    bstop = float([counter[6:] for counter in counters if counter.startswith('bstop=')][0])
    Omron = float([counter[6:] for counter in counters if counter.startswith('Omron=')][0])
    Temperature = float([counter[5:] for counter in counters if counter.startswith('TEMP=')][0])
    ROI1 = float([counter[6:] for counter in counters if counter.startswith('ROI1=')][0])
    ROI2 = float([counter[5:] for counter in counters if counter.startswith('ROI2=')][0])
    ROI3 = float([counter[5:] for counter in counters if counter.startswith('ROI3=')][0])
    ROI4 = float([counter[5:] for counter in counters if counter.startswith('ROI4=')][0])
    ROI5 = float([counter[5:] for counter in counters if counter.startswith('ROI5=')][0])
    ROI6 = float([counter[5:] for counter in counters if counter.startswith('ROI6=')][0])
    ROI7 = float([counter[5:] for counter in counters if counter.startswith('ROI7=')][0])
    ROI8 = float([counter[5:] for counter in counters if counter.startswith('ROI8=')][0])
    ROI9 = float([counter[5:] for counter in counters if counter.startswith('ROI9=')][0])
    ROI10 = float([counter[6:] for counter in counters if counter.startswith('ROI10=')][0])

    # parsing the coordinates
    motors = data[10].split(', ')
    plate_x = [motor[8:] for motor in motors if motor.startswith('plate_x')][0]
    if 'e' in plate_x:
        plate_x = 0.0
    else:
        plate_x = float(plate_x)
    plate_y = [motor[8:] for motor in motors if motor.startswith('plate_y')][0]
    if 'e' in plate_y:
        plate_y = 0.0
    else:
        plate_y = float(plate_y)


    metadata = [['plate_x', plate_x], ['plate_y', plate_y], ['I0', I0], ['I1', I1], ['bstop', bstop], ['Omron', Omron],
                ['Temperature', Temperature],['ROI1', ROI1],['ROI2', ROI2], ['ROI3', ROI3], ['ROI4', ROI4], ['ROI5', ROI5],
                ['ROI6', ROI6], ['ROI7', ROI7], ['ROI8', ROI8], ['ROI9', ROI9], ['ROI10', ROI10]]

    return metadata
#
# filepath = 'C:\Research_FangRen\Data\Apr2016\Jan_samples\Sample1\\Sample1_30x30_t60_0003.tif'
#
# metadata = extract_metadata(filepath)


