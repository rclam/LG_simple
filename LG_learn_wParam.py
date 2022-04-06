#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: rclam
"""

import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import tensorflow.keras.metrics

from sklearn.preprocessing import MinMaxScaler #, StandardScaler


import matplotlib.pyplot as plt
import sklearn.metrics

from tensorflow.keras.optimizers import Adam, SGD, Nadam, RMSprop

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   Load Data and Preprocess
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
scaler = MinMaxScaler(feature_range=(-1,1))

print('Loading temperature array...')
a_T = np.load("a_temp.npy")
print('Scaling data...')
a_T_sc = scaler.fit_transform(a_T.reshape(-1, a_T.shape[-1])).reshape(a_T.shape)


print('\nLoading parameters array...')
a_par = np.load("a_par_all.npy")
a_par_sc = scaler.fit_transform(a_par)
# a_par = a_par[:-1,:]   # temporary!!! make same size as a_T until remove proper file

# X = a_T_sc[:,:-1,:]      # Take only first 18 time step for training
# y = a_T_sc[:,-1,:]       # Target label = last time step




print('\nAdding param data into X...')
# X = X_new[:,:-1,:]      # Take only first 18 time step for training
X= np.concatenate((a_par_sc[:,None,:].repeat(18,1),a_T_sc[:,:-1,:]),axis=2)
print('\nStoring last time step data into y...')
y = a_T_sc[:,-1,:]       # Target label = last time step

# X = data[:,:-1,:]      # Take only first 18 time step for training
# y = data[:,-1,:]       # Target label = last time step

print('\nSeparating data into training and testing...')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)


nTimeSteps = len(X[0])
iLength = len(y[0])
nSequences = len(y)
# print('\nNo. training data (how many sequences): ', nSequences)
# print('No. time steps per sequence: ', nTimeSteps)
# print('No. data points per time step: ', iLength)

print('\n\n')
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   Define LSTM
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

input_shapes = ( nTimeSteps, iLength+5)

# define LSTM configuration
n_epoch =  100        #1200
n_hidden = 2 #2
f_dropout = 0.05


# build/create LSTM
model = Sequential()
model.add(LSTM(units=n_hidden, 
                return_sequences=True,
                input_shape=input_shapes, activation = 'sigmoid'))
model.add(Dropout(f_dropout))
for s_bool in [True, False]:
    model.add(LSTM(units = 2, return_sequences=s_bool))

# add fully connected output layer
model.add( Dense(units=12288))

# eta = 0.01
# opt = SGD(lr=eta)
# model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])

opt = Adam(learning_rate=0.0001)   #0.03
model.compile(loss='mse', optimizer=opt, 
              metrics=[tensorflow.keras.metrics.Accuracy()])

# loss = 'categorical_crossentropy'; 'mean_squared_error'
# optimizer = Adam; Nadam; SGD; RMSprop;


# check size of ANN
# model.summary()
# print(model.summary())




#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   Train LSTM
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

batch_size=16
# dResult = model.fit(X_train, y_train, 
#                     batch_size=batch_size,
#                     epochs=n_epoch,
#                     validation_split=0.3,
#                     shuffle=(False)).history

dResult = model.fit(X_train, y_train, 
                    batch_size=batch_size,
                    epochs=n_epoch,
                    validation_split=0.3,
                    shuffle=(True)).history

y_pred = model.predict(X_test)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   Evaluate LSTM
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
R2 = sklearn.metrics.r2_score(y_test, y_pred)
MSE = sklearn.metrics.mean_squared_error(y_test, y_pred, squared=(True))
RMSE = sklearn.metrics.mean_squared_error(y_test, y_pred, squared=(False))

print('\nR2 score: ', R2)
print('Mean squared Error:', MSE)
print('Root mean squared Error:', RMSE)

# print(dResult.keys())

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   Plot
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
plt.figure(1)
plt.plot(dResult['loss'],label='train', linestyle=':',lw=4)
plt.plot(dResult['val_loss'],label='val.', linestyle='-')
# plt.xlabel('epochs')
plt.ylabel('Loss', fontsize=24)
plt.yticks(fontsize=20)
plt.xticks(fontsize=16)
plt.legend( fontsize=12, loc='upper right')
plt.tight_layout()
plt.show()



# plt.plot(dResult['accuracy'], label='train', linestyle=':',lw=4)
# plt.plot(dResult['val_accuracy'],label='val.', linestyle='-')
# plt.xlabel('epochs', fontsize=20)
# plt.ylabel('Accuracy',fontsize=24)
# plt.yticks(fontsize=20)
# plt.xticks(fontsize=16)
# # plt.legend()
# plt.tight_layout()
# plt.show()


plt.figure(2)
plt.contourf(y_pred)
plt.colorbar(label='T')
plt.xlabel('point')
plt.ylabel('model')
plt.title('predicted')
plt.show()

plt.figure(3)
plt.contourf(y_test)
plt.colorbar(label='T')
plt.xlabel('point')
plt.ylabel('model')
plt.title('true')
plt.show()

#~~~~~~~~~

pick = 930

plt.figure(4)
# plt.contourf(y_test)
# plt.colorbar(label='T')
plt.plot(y_pred[pick,:],'.')
# plt.xlabel('point')
# plt.ylabel('model')
plt.title('predicted: model %i'%(pick))
plt.show()

plt.figure(5)
# plt.contourf(y_test)
# plt.colorbar(label='T')
plt.plot(y_test[pick,:],'.')
# plt.xlabel('point')
# plt.ylabel('model')
plt.title('true: model %i'%(pick))
plt.show()



y_pred_rs = y_pred[3].reshape(64, 192)
y_test_rs = y_test[3].reshape(64, 192)
plt.figure(6)
plt.contourf(y_pred_rs, cmap='coolwarm')
plt.colorbar(label='T')
# plt.plot(y_pred_rs,'.')
# plt.xlabel('point')
# plt.ylabel('model')
plt.title('predicted: model %i'%(pick))
plt.show()

plt.figure(7)
plt.contourf(y_test_rs, cmap='coolwarm')  # 'bwr', 'coolwarm', 'jet'
plt.colorbar(label='T')
# plt.plot(y_test_rs,'.')
# plt.xlabel('point')
# plt.ylabel('model')
plt.title('true: model %i'%(pick))
plt.show()

# ) Plot histogram of misfit
plt.figure(8)
residual_test = y_pred - y_test
plt.hist(residual_test[pick])
plt.ylabel('frequency' )
plt.xlabel('misfit for sequence y_{test}[%i]'%(pick))
plt.show()


print(model.summary())
print('\nR2 score: ', R2)
print('Mean squared Error:', MSE)
print('Root mean squared Error:', RMSE)

