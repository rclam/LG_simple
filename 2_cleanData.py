import os
import numpy as np
from paraview.simple import *
import clean2grid

config = 0

for c in range(695):
    step=0
    for ts in range(19):
        print('\nc: ', config)
        print('ts: ', step)
        clean2grid.clean2grid(config, step)
        step += 1
    config +=1
