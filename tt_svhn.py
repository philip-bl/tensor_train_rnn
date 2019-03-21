import numpy as np
import datetime
import scipy.io as sio
import os

from keras.layers import Input, Dense
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import *
from keras.datasets import mnist

from third_parties.TT_RNN.TTLayer import TT_Layer

from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score

np.random.seed(11111986)

# run `load_svhn.sh` to download data
path_to_data = './' # change if data is in a different folder


train = sio.loadmat('{path_to_data}train_32x32.mat')
X_train = np.rollaxis(train['X'], 3)
y_train = np.mod(np.rollaxis(train['y'], 1)[0], 10)

test = sio.loadmat('{path_to_data}test_32x32.mat')
X_test = np.rollaxis(test['X'], 3)
y_test = np.mod(np.rollaxis(test['y'], 1)[0], 10)

X_train = X_train.astype('float32')
Y_train = to_categorical(y_train.astype('int32'), 10)
X_test = X_test.astype('float32')
Y_test = to_categorical(y_test.astype('int32'), 10)

X_train /= 255
X_test /= 255

X_train = X_train.reshape((X_train.shape[0], 32*32 * 3))
X_test = X_test.reshape((X_test.shape[0], 32*32 * 3))

n_epochs = 100

# ----- with TTLayer -----
input = Input(shape=(32 * 32 * 3,))
h1 = TT_Layer(tt_input_shape=[4, 8, 8, 12], tt_output_shape=[4, 8, 8, 12], tt_ranks=[1, 3, 3, 3, 1], activation='relu', kernel_regularizer=l2(5e-4), debug=False)(input)
output = Dense(output_dim=10, activation='softmax', kernel_regularizer=l2(1e-3))(h1)

model = Model(input=input, output=output)
model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])


start1 = datetime.datetime.now()
history1 = model.fit(x=X_train, y=Y_train, verbose=2, epochs=n_epochs, batch_size=128, validation_split=0.2)
stop1 = datetime.datetime.now()
print(f'Time to train: {stop1 - start1}')
model.save('model_tt.h5')

print('TT qualities:')
Y_hat = model.predict(X_train)
print(accuracy_score(Y_train, np.round(Y_hat)))
Y_pred = model.predict(X_test)
print(accuracy_score(Y_test, np.round(Y_pred)))

# ----- with Dense layer -----

input = Input(shape=(32*32*3,))
h1 = Dense(output_dim=32*32*3, activation='relu', kernel_regularizer=l2(5e-4))(input)
output = Dense(output_dim=10, activation='softmax', kernel_regularizer=l2(1e-3))(h1)
model2 = Model(input=input, output=output)
model2.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])


start2 = datetime.datetime.now()
history2 = model2.fit(x=X_train, y=Y_train, verbose=2, nb_epoch=n_epochs, batch_size=128, validation_split=0.2)
stop2 = datetime.datetime.now()
print(f'Time to train: {stop2 - start2}')
model2.save('model_dense.h5')

print('Dense qualities:')
Y_hat = model2.predict(X_train)
print(accuracy_score(Y_train, np.round(Y_hat)))
Y_pred = model2.predict(X_test)
print(accuracy_score(Y_test, np.round(Y_pred)))

# ----- plot the results -----
losses = [history1.history['loss'], history1.history['val_loss'], history2.history['loss'], history2.history['val_loss']]
accs = [history1.history['acc'], history1.history['val_acc'], history2.history['acc'], history2.history['val_acc']]
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

metrics = {"Loss": losses, "Accuracy": accs}
for ax, metric in zip(axes, metrics):
    ax.set_title(metric)
    m = metrics[metric]
    ax.plot(range(n_epochs), m[0], label='TT Train')
    ax.plot(range(n_epochs), m[1], label='TT Valid')
    ax.plot(range(n_epochs), m[2], label='Dense Train')
    ax.plot(range(n_epochs), m[3], label='Dense Valid')
    ax.set_xlabel("Number of epochs")
    ax.legend()
plt.savefig('tt_svhn.png')
