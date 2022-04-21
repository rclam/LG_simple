#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 13:03:59 2022

@author: rclam
"""
from paraview.simple import *

def clean2grid(config, timeStep):
    
    name = "solution-0{0:04d}.0000.vtu".format(timeStep)
    print('name:', name)
    outerFolder = "output_LG_single_LG_sConfig_{0:04d}".format(config)
    print('folder: ', outerFolder)
    INdataDirect = '/Volumes/RC_LaCie/projects/Liu_Gurnis_simple/2_outputs/'+outerFolder+'/solution/'+name
    
    
    OUTdataDirect= "/Volumes/RC_LaCie/projects/Liu_Gurnis_simple/2_outputs_clean/config{0:04d}/".format(config)
    OUTfname1= "clean_config{0:04d}_".format(config)
    OUTfname = OUTfname1 + name 
    print('out name: ', OUTfname)
    OUTsave = OUTdataDirect+OUTfname

    #### disable automatic camera reset on 'Show'
    paraview.simple._DisableFirstRenderCameraReset()
    
    # create a new 'XML Unstructured Grid Reader'
    solution000000000vtu = XMLUnstructuredGridReader(registrationName=name, FileName=[INdataDirect])
    # solution000000000vtu = XMLUnstructuredGridReader(registrationName='solution-00000.0000.vtu', FileName=['/home/pv-user/data/output_LG_single_LG_sConfig_0000/solution/solution-00000.pvtu'])
    solution000000000vtu.PointArrayStatus = ['velocity', 'p', 'T', 'density', 'thermal_expansivity', 'specific_heat', 'viscosity']
    
    # Properties modified on solution000000000vtu
    solution000000000vtu.TimeArray = 'None'
    
    # get active view
    spreadSheetView1 = GetActiveViewOrCreate('SpreadSheetView')
    
    # show data in view
    solution000000000vtuDisplay = Show(solution000000000vtu, spreadSheetView1, 'SpreadSheetRepresentation')
    
    # update the view to ensure updated data information
    spreadSheetView1.Update()
    
    # create a new 'Clean to Grid'
    cleantoGrid1 = CleantoGrid(registrationName='CleantoGrid1', Input=solution000000000vtu)
    
    # show data in view
    cleantoGrid1Display = Show(cleantoGrid1, spreadSheetView1, 'SpreadSheetRepresentation')
    
    # hide data in view
    Hide(solution000000000vtu, spreadSheetView1)
    
    # update the view to ensure updated data information
    spreadSheetView1.Update()
    
    # save data
    SaveData( OUTsave, proxy=cleantoGrid1, PointDataArrays=['T', 'density', 'p', 'specific_heat', 'thermal_expansivity', 'velocity', 'viscosity'],
        FieldDataArrays=['TIME'],
        DataMode='Ascii')
