import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv2D, MaxPool2D, RepeatVector, Flatten, TimeDistributed
import tensorflow.keras.metrics

from sklearn.preprocessing import MinMaxScaler #, StandardScaler


import matplotlib.pyplot as plt
import sklearn.metrics

from tensorflow.keras.optimizers import Adam #, SGD, Nadam, RMSprop
import time
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   Load Data and Preprocess
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
scaler = MinMaxScaler(feature_range=(0,1))  #(0,1) for sigmoid, (-1,1) for tanh, (0, inf) for relu

print('\nload point coordinate data...')
x_loc = np.load("x_loc_clean.npy")
y_loc = np.load("y_loc_clean.npy")

print('scale x- and y-coord....') # this is for plotting purposes
x_loc_sc = scaler.fit_transform(x_loc.reshape(-1,1))
y_loc_sc = scaler.fit_transform(y_loc.reshape(-1,1))


# ~~~~~~~~~ Data (mult. options) ~~~~~~~~~
# # use only Temperature distributions
# print('\nLoading temperature data array...')
# a_T  = np.load("a_T_clean.npy")
# X = a_T[:,:-1,:]
# X = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
# X = X.reshape(3124, 18, 3201, 1)

# use 5 parameters and temperature dist. (both scaled), x- and y-coord (true)
# print('\nLoading scaled (all param. except coord) X data array...')
# X  = np.load("a_par_loc_T_sc.npy")

# all 8 parameters scaled
print('\nLoading scaled (all param. INCL. coord) X data array...')
X  = np.load("a_par_loc_T_sc_all.npy")
# X  = np.load("a_par_loc_sc_T_NOTsc.npy")
print('X shape: ', X.shape)


# print('Reshaping X data array...')
# X = np.reshape( X, (3124, 18, 3201*8))
# print('X shape: ', X.shape)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print('\nLoading scaled y data (last time step Temperature data)...')
y = np.load("a_y_sc.npy")
# y = np.load("a_y_NOTsc.npy")
print('y shape: ', y.shape)



print('\nSeparating data into training and testing...')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)


nTimeSteps = len(X[0])
iLength = len(y[0])
nSequences = len(y)
# print('\nNo. training data (how many sequences): ', nSequences)
# print('No. time steps per sequence: ', nTimeSteps)
# # print('No. data points per time step: ', iLength)

print('\n\n')


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   Define LSTM
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# define LSTM configuration
n_epoch =  100        #1200
# n_hidden = 30 #2, 50
f_dropout = 0.05 #0.05


# build/create LSTM

# input_shapes = ( nTimeSteps, iLength*8)
# print('input shape: ', input_shapes)



model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(8, 8), 
                 strides=(3,3), padding='same',
                 data_format='channels_last', 
                 name='conv_1', activation='relu'))   #32, k=(8,8)

model.add(MaxPool2D(pool_size=(2, 2), name='pool_1',
                    strides=2))

model.add(Conv2D(filters=64, kernel_size=(5,5), 
                 strides=(2, 2), padding='same',
                 name='conv_2', activation='relu'))   #64, strides=(2, 2)

model.add(MaxPool2D(pool_size=(2, 2), name='pool_2',
                    strides=2))


# model.add(TimeDistributed(Flatten(input_shape=(3201,8))))
model.add(TimeDistributed(Flatten(), name='time_dist_1'))

# model.add(LSTM(units=n_hidden, 
#                 return_sequences=True,
#                 input_shape=input_shapes, activation = 'sigmoid'))

model.add(LSTM(units=128, return_sequences=True, 
               activation = 'relu', name='lstm_1'))

model.add(LSTM(64, return_sequences=False, name='lstm_2'))

model.add(RepeatVector(18))

model.add(Dropout(f_dropout, name='dropout_1'))

# for s_bool in [True, False]:
#     model.add(LSTM(units = 128, return_sequences=s_bool, name='lstm_%s'%(s_bool)))

model.add(LSTM(units = 128, return_sequences=True, name='lstm_3'))
model.add(LSTM(units = 128, return_sequences=False, name='lstm_4'))


# add fully connected output layer
model.add( Dense(units=3201, name='fc_1'))

# model.build(input_shape=(None, 18, 3201, 8))

eta = 0.0002

opt = Adam(learning_rate=eta)   #0.03, 0.0001
model.compile(loss='mse', optimizer=opt, 
              metrics=[tensorflow.keras.metrics.Accuracy()])


# model.summary()



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   Train LSTM
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
t_start = time.time()
batch_size=16


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
t_end = time.time()
t_elapse = abs(t_start - t_end)

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
plt.title('Eta = %.5f'%(eta))
plt.tight_layout()
plt.show()

plt.figure(2)
plt.plot(dResult['accuracy'], label='train', linestyle=':',lw=4)
plt.plot(dResult['val_accuracy'],label='val.', linestyle='-')
plt.xlabel('epochs', fontsize=20)
plt.ylabel('Accuracy',fontsize=24)
plt.yticks(fontsize=20)
plt.xticks(fontsize=16)
# plt.legend()
plt.tight_layout()
plt.show()

plt.figure(3)
plt.subplot(211)
plt.plot(dResult['loss'],label='train', linestyle=':',lw=4)
plt.plot(dResult['val_loss'],label='val.', linestyle='-')
# plt.xlabel('epochs')
plt.ylabel('Loss', fontsize=24)
plt.yticks(fontsize=20)
plt.xticks(fontsize=16)
plt.legend( fontsize=12, loc='upper right')
plt.title('Eta = %.5f'%(eta))
plt.tight_layout()
# plt.show()

plt.subplot(212)
plt.plot(dResult['accuracy'], label='train', linestyle=':',lw=4)
plt.plot(dResult['val_accuracy'],label='val.', linestyle='-')
plt.xlabel('epochs', fontsize=20)
plt.ylabel('Accuracy',fontsize=24)
plt.yticks(fontsize=20)
plt.xticks(fontsize=16)
# plt.legend()
plt.tight_layout()
plt.show()


# plt.figure(4)
# vmin, vmax = 0, 1
# n = 7
# levels = np.linspace(vmin, vmax, n+1)
# plt.contourf(y_pred, levels=levels)
# plt.colorbar(label='T')
# plt.xlabel('point')
# plt.ylabel('model')
# plt.title('predicted')
# plt.show()

# plt.figure(5)
# plt.contourf(y_test, levels=levels)
# plt.colorbar(label='T')
# plt.xlabel('point')
# plt.ylabel('model')
# plt.title('true')
# plt.show()

#~~~~~~~~~

pick = 100


plt.figure(6)
plt.scatter(x_loc_sc, y_loc_sc, c=y_test[pick], cmap='coolwarm', vmin=0, vmax=1)
# plt.colorbar(label='T (scaled \u00b0K)')
plt.colorbar(label='T')
plt.title('true: model %i'%(pick))
plt.show()


plt.figure(7)
plt.scatter(x_loc_sc, y_loc_sc, c=y_pred[pick], cmap='coolwarm', vmin=0, vmax=1)
# plt.colorbar(label='T (\u00b0)')
plt.colorbar(label='T')
plt.title('predicted: model %i'%(pick))
plt.show()

# ) Plot histogram of misfit
plt.figure(8)
residual_test = y_pred - y_test
plt.hist(residual_test[pick], bins=25)
plt.ylabel('frequency' )
plt.xlabel('misfit for sequence: y test [%i]'%(pick))
plt.show()

plt.figure(9)
residual_test = abs(y_pred - y_test)
plt.scatter(x_loc_sc, y_loc_sc, c=residual_test[pick], cmap='gray_r')
# plt.colorbar(label='T (scaled \u00b0K)')
plt.colorbar(label='Abs. Error')
plt.title('residuals: model %i'%(pick))
plt.show()


print(model.summary())
print('\nR2 score: ', R2)
print('Mean squared Error:', MSE)
print('Root mean squared Error:', RMSE)

print('\ntime to train: ', t_elapse/60)
