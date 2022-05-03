#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 11:23:37 2022

@author: rclam
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler


scaler = MinMaxScaler(feature_range=(0,1))  #(0,1) for sigmoid, (-1,1) for tanh, (0, inf) for relu

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
inp_locs = np.vstack((a_x_loc_sc, a_y_loc_sc)).T


"""
new_inp = np.ones((8, 3201))
new_inp[0,:] = a_x_loc
new_inp[1,:] = a_y_loc
new_inp[2,:] = a_par[0,0].repeat(3201)
new_inp[3,:] = a_par[0,1].repeat(3201)
new_inp[4,:] = a_par[0,2].repeat(3201)
new_inp[5,:] = a_par[0,3].repeat(3201)
new_inp[6,:] = a_par[0,4].repeat(3201)
new_inp[7,:] = a_T[0,0,:]

new_inp_trial = np.ones((18,8,3201))
"""


a_par_rep = a_par_sc[:,None,:].repeat(3201,1)
inp_locs_rep = np.zeros((3124, 3201, 2))
# for i in range(3201):
#     inp_locs_rep[:,i,:] = inp_locs[i,:]

a_new_par = np.zeros((3124, 3201, 7))


for i in range( 3124):
    a_new_par[i,:,0] = a_par_sc[i,0].repeat(3201)
    a_new_par[i,:,1] = a_par_sc[i,1].repeat(3201)
    a_new_par[i,:,2] = a_par_sc[i,2].repeat(3201)
    a_new_par[i,:,3] = a_par_sc[i,3].repeat(3201)
    a_new_par[i,:,4] = a_par_sc[i,4].repeat(3201)


a_new_par[:,:,5] = inp_locs[:,0]
a_new_par[:,:,6] = inp_locs[:,1]

# np.save('a_par_w_locs_sc.npy', a_new_par)

print('Loading temperature array...')
a_T = np.load("a_T_clean.npy")
# print('Scaling data...')
# a_T_sc = scaler.fit_transform(a_T.reshape(-1, a_T.shape[-1])).reshape(a_T.shape)


# print('\nLoading parameters array...')
# a_par = np.load("a_par_w_locs.npy")


print('\nExpanding parameters array for all time steps...')
# a_par_expand = a_par[:,None,:].repeat(18,1)
a_par_expand = a_new_par[:,None,:].repeat(18,1)


print('\nCreating X array...')
X = np.zeros((3124, 18, 3201, 8))


a_T_x = a_T[:,:-1,:]
a_T_trial = a_T_x.reshape(3124, 18, 3201, 1)
X = np.concatenate((a_par_expand, a_T_trial), axis=3)

print('\nStoring last time step T data into y...')
# y = a_T_sc[:,-1,:]       # Target label = last time step
y = a_T[:,-1,:]       # Target label = last time step

# np.save('a_par_loc_sc_T_NOTsc.npy', X)
# np.save('a_y_NOTsc.npy', y)