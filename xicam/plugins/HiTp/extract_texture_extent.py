# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 10:52:16 2016

@author: fangren, Apurva Mehta
"""
import numpy as np

def extract_texture_extent(Qlist_texture, texture):
    texture = np.array(texture)
    texture = texture **2
    SqrSum = np.nansum(texture)
    NormSqrSum = SqrSum/float(len(Qlist_texture))
    newRow = [['texture_sum', NormSqrSum]]
    return newRow