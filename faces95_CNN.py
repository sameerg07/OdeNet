from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization, MaxPooling2D,Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K



import os
import numpy as np
np.random.seed(1337)  # for reproducibility

# from keras.datasets import mnist
# import load_faces95_data
import pickle
from keras.models import Sequential
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import LearningRateScheduler
from keras.utils import np_utils
K.set_image_data_format('channels_first')


from binary_ops import binary_tanh as binary_tanh_op


H = 1.
kernel_lr_multiplier = 'Glorot'

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

# nn
# batch_size = 50
batch_size = 10
epochs = 50
nb_channel = 1
img_rows = 71 
img_cols = 64
# img_rows = 64 
# img_cols = 64
nb_filters = 32 
nb_conv = 3
nb_pool = 2
nb_hid = 128
nb_classes = 72
use_bias = False

# learning rate schedule
lr_start = 1e-3
lr_end = 1e-4
lr_decay = (lr_end / lr_start)**(1. / epochs)

# BN
epsilon = 1e-6
momentum = 0.9

# dropout
p1 = 0.25
p2 = 0.5
# data needs to be a tuple with two elements.
# the data, shuffled and split between train and test sets
# (X_train, y_train), (X_test, y_test) = load_faces95_data.load_data()
# pickledDataset = open("faces95dataWithPCA.pickle","rb")
pickledDataset = open("faces95dataWithImgSalMBD.pickle","rb")
data = pickle.load(pickledDataset)
(X_train, y_train), (X_test, y_test) = data['res']
print("Dataset has been loaded")

X_train = X_train.reshape(1224, 1, 71, 64)
X_test = X_test.reshape(216, 1, 71, 64)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes) * 2 - 1 # -1 or 1 for hinge loss
Y_test = np_utils.to_categorical(y_test, nb_classes) * 2 - 1

model = Sequential()
model.add(Conv2D(256, kernel_size=(3, 3),   
                     padding='same', use_bias=use_bias, name='conv4'))
# remove 3 lines below if needed
# model.add(Conv2D(256, kernel_size=(3, 3),   
#                      padding='same', use_bias=use_bias, name='conv4_a'))
# model.add(Conv2D(256, kernel_size=(3, 3),   
#                      padding='same', use_bias=use_bias, name='conv4_b'))
# model.add(Conv2D(256, kernel_size=(3, 3),   
#                      padding='same', use_bias=use_bias, name='conv4_c'))
model.add(MaxPooling2D(pool_size=(2, 2), name='pool4'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='bn4'))
model.add(Activation('selu', name='act4'))
model.add(Flatten())
# dense1
model.add(Dense(1024,   use_bias=use_bias, name='dense5'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn5'))
model.add(Activation('selu', name='act5'))
# dense2
model.add(Dense(nb_classes,   use_bias=use_bias, name='dense6'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn6'))

opt = Adam(lr=lr_start) 
model.compile(loss='squared_hinge', optimizer=opt, metrics=['acc'])

lr_scheduler = LearningRateScheduler(lambda e: lr_start * lr_decay ** e)
history = model.fit(X_train, Y_train,
                    batch_size=batch_size, epochs=epochs,
                    verbose=1, validation_data=(X_test, Y_test),
                    callbacks=[lr_scheduler])

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])