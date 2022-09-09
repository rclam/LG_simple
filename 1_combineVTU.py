#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read in all npy files and save in one npy

@author: rclam
"""

import numpy as np
#from numpy import load
#from scipy.sparse import dok_matrix

import os

data_dir = "/Volumes/RC_LaCie/projects/Liu_Gurnis_simple/2_outputs_clean/group1"
a_combine = []

for f in os.listdir(data_dir):

    if f.startswith("a_T_clean"):
        filename = f
        # print(filename)
        a_combine.append((filename))


n1, n2 = 19, 3201 #21, 6912  # n1=timesteps, n2=points of data

combined_data = np.zeros((len(a_combine), n1, n2))
print('combined: ',combined_data.shape)

for i, fn in enumerate(a_combine):
    print(i, fn)
    print(np.load(fn).shape)
    combined_data[i, :, :] = np.load(fn)

np.save('a_T_clean_1.npy',combined_data)
