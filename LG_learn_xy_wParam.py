import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv2D, MaxPool2D, Flatten, TimeDistributed
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
print('X shape: ', X.shape)


# print('Reshaping X data array...')
# X = np.reshape( X, (3124, 18, 3201*8))
# print('X shape: ', X.shape)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print('\nLoading scaled y data (last time step Temperature data)...')
y = np.load("a_y_sc.npy")
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
n_epoch =  20        #1200
n_hidden = 50 #2
f_dropout = 0.05 #0.05


# build/create LSTM

input_shapes = ( nTimeSteps, iLength*8)
# print('input shape: ', input_shapes)


model = Sequential()
model.add(TimeDistributed(Flatten(input_shape=(3201,8))))
# model.add(LSTM(units=n_hidden, 
#                 return_sequences=True,
#                 input_shape=input_shapes, activation = 'sigmoid'))
model.add(LSTM(units=n_hidden, 
                return_sequences=True, activation = 'sigmoid'))
model.add(Dropout(f_dropout))
for s_bool in [True, False]:
    model.add(LSTM(units = 2, return_sequences=s_bool))

# add fully connected output layer
model.add( Dense(units=3201))

eta = 0.0001

opt = Adam(learning_rate=eta)   #0.03, 0.0001
model.compile(loss='mse', optimizer=opt, 
              metrics=[tensorflow.keras.metrics.Accuracy()])

"""
model = Sequential()
model.add(Conv2D(filters=8, kernel_size=(7,7), 
                 strides=(1,1), padding='same',
                 data_format='channels_last', 
                 name='conv_1', activation='relu'))

model.add(MaxPool2D(pool_size=(2, 2), name='pool_1',
                    strides=2))

model.add(Conv2D(filters=16, kernel_size=(5,5), 
                  strides=(1,1), padding='same',
                  name='conv_2', activation='relu'))

# model.add(MaxPool2D(pool_size=(1, 1), name='pool_2'))
model.add(MaxPool2D(pool_size=(2, 2), name='pool_2'))

model.compute_output_shape(input_shape=X.shape)

model.add(Flatten())
model.compute_output_shape(input_shape=X.shape)

model.add(Dense(units=9306, name='fc_1',
                activation='relu'))

# model.add(Dropout(rate=0.05))

model.add(Dense(units=3201, name='fc_2',
                activation='softmax'))

# model.build(input_shape=(None, 18, 3201, 8))
model.build(input_shape=(None, 18, 3201, 1))
eta = 0.0001
opt = Adam(learning_rate=eta)   #0.03, 0.0001
model.compile(loss='mse', optimizer=opt, 
              metrics=[tensorflow.keras.metrics.Accuracy()])


model.summary()

"""
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   Train LSTM
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
t_start = time.time()
batch_size=16


dResult = model.fit(X_train, y_train, 
                    batch_size=batch_size,
                    epochs=n_epoch,
                    validation_split=0.3,
                    shuffle=(False)).history

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

# plt.figure(3)
# plt.contourf(y_test)
# plt.colorbar(label='T')
# plt.xlabel('point')
# plt.ylabel('model')
# plt.title('true')
# plt.show()

#~~~~~~~~~

pick = 100


plt.figure(4)
plt.scatter(x_loc_sc, y_loc_sc, c=y_test[pick], cmap='coolwarm', vmin=0, vmax=1)
# plt.colorbar(label='T (scaled \u00b0K)')
plt.colorbar(label='T')
plt.title('true: model %i'%(pick))
plt.show()


plt.figure(5)
plt.scatter(x_loc_sc, y_loc_sc, c=y_pred[pick], cmap='coolwarm', vmin=0, vmax=1)
# plt.colorbar(label='T (\u00b0)')
plt.colorbar(label='T')
plt.title('predicted: model %i'%(pick))
plt.show()

# # ) Plot histogram of misfit
# plt.figure(6)
# residual_test = y_pred - y_test
# plt.hist(residual_test[pick])
# plt.ylabel('frequency' )
# plt.xlabel('misfit for sequence y_{test}[%i]'%(pick))
# plt.show()


print(model.summary())
print('\nR2 score: ', R2)
print('Mean squared Error:', MSE)
print('Root mean squared Error:', RMSE)

print('\ntime to train: ', t_elapse/60)
