#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 11:54:21 2022

@author: rclam
"""

import os
import numpy as np

from vtk import vtkXMLUnstructuredGridReader
from vtk.util.numpy_support import vtk_to_numpy




# empty list of time steps


for numb in range(625):
    a_T_clean = []
    print(type(a_T_clean))
    numb = 0 + numb


    directory = '/Volumes/RC_LaCie/projects/Liu_Gurnis_simple/2_outputs_clean/config{0:04d}'.format(numb)
    print(directory)


    for f in os.listdir(directory):
        if f.endswith(".vtu"):
            filename = directory+'/'+f
            print(filename)
            reader = vtkXMLUnstructuredGridReader()
            reader.SetFileName(filename)
            reader.Update()
            data = reader.GetOutput()
            T_hist = vtk_to_numpy(data.GetPointData().GetVectors('T'))

            a_T_clean.append((T_hist))

    #np.save('a_T_0050.npy',a_T, allow_pickle=True)
    np.save('a_T_clean_{0:04d}.npy'.format(numb),a_T_clean, allow_pickle=True)

    #print(a_T[0])

