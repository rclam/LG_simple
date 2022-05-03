#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 11:23:37 2022

@author: rclam
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler


scaler = MinMaxScaler(feature_range=(0,1))

print('Loading x_locations array...')
a_x_loc = np.load("x_loc_clean.npy")
a_x_loc_sc = scaler.fit_transform(a_x_loc.reshape(-1,1))

print('Loading y_locations array...')
a_y_loc = np.load("y_loc_clean.npy")
a_y_loc_sc = scaler.fit_transform(a_y_loc.reshape(-1,1))


inp_locs = np.vstack((a_x_loc, a_y_loc))



print('\nLoading parameters array...')
a_par = np.load("a_par_all.npy")
a_par_sc = scaler.fit_transform(a_par)

# inp_locs = np.vstack((a_x_loc, a_y_loc)).T
inp_locs = np.vstack((a_x_loc_sc, a_y_loc_sc)).T # combine co-ordinate data (here scaled)



a_par_rep = a_par_sc[:,None,:].repeat(3201,1) # repeat parameters (scaled) for each of the 3201 points

a_new_par = np.zeros((3124, 3201, 7)) # initialize new data array for 5 parameters columns and 2 co-ordinate columns per data point per time step

# fill new data array with original parameters
for i in range( 3124):
    a_new_par[i,:,0] = a_par_sc[i,0].repeat(3201)
    a_new_par[i,:,1] = a_par_sc[i,1].repeat(3201)
    a_new_par[i,:,2] = a_par_sc[i,2].repeat(3201)
    a_new_par[i,:,3] = a_par_sc[i,3].repeat(3201)
    a_new_par[i,:,4] = a_par_sc[i,4].repeat(3201)

# fill new data array with co-ordinate data
a_new_par[:,:,5] = inp_locs[:,0]
a_new_par[:,:,6] = inp_locs[:,1]

# np.save('a_par_w_locs_sc.npy', a_new_par). # can save as is if desire

print('Loading temperature array...')
a_T = np.load("a_T_clean.npy")
print('Scaling data...')
a_T_sc = scaler.fit_transform(a_T.reshape(-1, a_T.shape[-1])).reshape(a_T.shape)


# print('\nLoading parameters array...') # if already saved new parameter array (without temperature data), can load here
# a_par = np.load("a_par_w_locs.npy")


print('\nExpanding parameters array for all time steps...')
# a_par_expand = a_par[:,None,:].repeat(18,1)
a_par_expand = a_new_par[:,None,:].repeat(18,1)


print('\nCreating X array...')     # initialize new X array
X = np.zeros((3124, 18, 3201, 8))


a_T_x = a_T_sc[:,:-1,:]                                 # keep all but last time step data
a_T_trial = a_T_x.reshape(3124, 18, 3201, 1)            # reshape for CNN input
X = np.concatenate((a_par_expand, a_T_trial), axis=3)   # combine parameter array and temperature array

print('\nStoring last time step T data into y...')
# y = a_T_sc[:,-1,:]       # Target label = last time step
y = a_T[:,-1,:]       # Target label = last time step

# np.save('a_par_loc_T_sc.npy', X) # save files
# np.save('a_y_sc.npy', y)
