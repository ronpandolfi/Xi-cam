"""
author: Fang Ren (SSRL)

5/25/2017
"""
from shutil import copyfile
import os.path
import time
import glob, os

src_folder = 'C:\\Research_FangRen\\Data\\SampleB2_23_200C'
dst_folder = 'C:\\Research_FangRen\\Data\\Sample_100'

os.chdir(src_folder)
for file in glob.glob("*.tif"):
    filename = os.path.basename(file)
    txt = file[:-3]+'txt'
    txtname = os.path.basename(txt)
    dst = os.path.join(dst_folder, filename)
    txtdst = os.path.join(dst_folder, txtname)
    copyfile(file, dst)
    copyfile(txt, txtdst)
    time.sleep(15)