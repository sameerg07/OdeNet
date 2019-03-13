'''Trains a simple xnor CNN on the MNIST dataset.
Modified from keras' examples/mnist_mlp.py
Gets to 98.18% test accuracy after 20 epochs using tensorflow backend
'''
#ORL
#Faces95
#Faces96
#Grimace

from __future__ import print_function
import os
import numpy as np
np.random.seed(1337)  # for reproducibility

# from keras.datasets import mnist
import load_grimace_data
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization, MaxPooling2D
from keras.layers import Flatten
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import LearningRateScheduler
from keras.utils import np_utils
import keras.backend as K
K.set_image_data_format('channels_first')


from binary_ops import binary_tanh as binary_tanh_op
from xnor_layers import XnorDense, XnorConv2D


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
nb_filters = 32 
nb_conv = 3
nb_pool = 2
nb_hid = 128
nb_classes = 18
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
# (X_train, y_train), (X_test, y_test) = load_grimace_data.load_data()

pickledDataset = open("grimacedataWithImgSalMBD.pickle","rb")
data = pickle.load(pickledDataset)
(X_train, y_train), (X_test, y_test) = data['res']
print("Dataset has been loaded")

X_train = X_train.reshape(306, 1, 71, 64)
X_test = X_test.reshape(54, 1, 71, 64)
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
# number is layers is number of Xnor*** and BatchNormalization
# conv1
model.add(XnorConv2D(128, kernel_size=(3, 3), input_shape=(nb_channel, img_rows, img_cols),
                     H=H, kernel_lr_multiplier=kernel_lr_multiplier, 
                     padding='same', use_bias=use_bias, name='conv1'))
# remove 2 lines below if needed
model.add(XnorConv2D(128, kernel_size=(3, 3), input_shape=(nb_channel, img_rows, img_cols),
                     H=H, kernel_lr_multiplier=kernel_lr_multiplier, 
                     padding='same', use_bias=use_bias, name='conv1_a'))
model.add(MaxPooling2D(pool_size=(2, 2), name='pool1'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='bn1'))
model.add(Activation('relu', name='act1'))

# conv2
model.add(XnorConv2D(128, kernel_size=(3, 3), H=H, kernel_lr_multiplier=kernel_lr_multiplier, 
                     padding='same', use_bias=use_bias, name='conv2'))
# remove 2 lines below if needed
model.add(XnorConv2D(128, kernel_size=(3, 3), H=H, kernel_lr_multiplier=kernel_lr_multiplier, 
                     padding='same', use_bias=use_bias, name='conv2_a'))
model.add(MaxPooling2D(pool_size=(2, 2), name='pool2'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='bn2'))
model.add(Activation('relu', name='act2'))

# conv3
model.add(XnorConv2D(256, kernel_size=(3, 3), H=H, kernel_lr_multiplier=kernel_lr_multiplier, 
                     padding='same', use_bias=use_bias, name='conv3'))
# remove 3 lines below if needed
model.add(XnorConv2D(256, kernel_size=(3, 3), H=H, kernel_lr_multiplier=kernel_lr_multiplier, 
                     padding='same', use_bias=use_bias, name='conv3_a'))
model.add(XnorConv2D(256, kernel_size=(3, 3), H=H, kernel_lr_multiplier=kernel_lr_multiplier, 
                     padding='same', use_bias=use_bias, name='conv3_b'))
model.add(XnorConv2D(256, kernel_size=(3, 3), H=H, kernel_lr_multiplier=kernel_lr_multiplier, 
                     padding='same', use_bias=use_bias, name='conv3_c'))
model.add(MaxPooling2D(pool_size=(2, 2), name='pool3'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='bn3'))
model.add(Activation('relu', name='act3'))

# conv4
model.add(XnorConv2D(256, kernel_size=(3, 3), H=H, kernel_lr_multiplier=kernel_lr_multiplier, 
                     padding='same', use_bias=use_bias, name='conv4'))
# remove 3 lines below if needed
model.add(XnorConv2D(256, kernel_size=(3, 3), H=H, kernel_lr_multiplier=kernel_lr_multiplier, 
                     padding='same', use_bias=use_bias, name='conv4_a'))
model.add(XnorConv2D(256, kernel_size=(3, 3), H=H, kernel_lr_multiplier=kernel_lr_multiplier, 
                     padding='same', use_bias=use_bias, name='conv4_b'))
model.add(XnorConv2D(256, kernel_size=(3, 3), H=H, kernel_lr_multiplier=kernel_lr_multiplier, 
                     padding='same', use_bias=use_bias, name='conv4_c'))
model.add(MaxPooling2D(pool_size=(2, 2), name='pool4'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='bn4'))
model.add(Activation('relu', name='act4'))
model.add(Flatten())
# dense1
model.add(XnorDense(1024, H=H, kernel_lr_multiplier=kernel_lr_multiplier, use_bias=use_bias, name='dense5'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn5'))
model.add(Activation('relu', name='act5'))
# dense2
model.add(XnorDense(nb_classes, H=H, kernel_lr_multiplier=kernel_lr_multiplier, use_bias=use_bias, name='dense6'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn6'))

opt = Adam(lr=lr_start) 
model.compile(loss='squared_hinge', optimizer=opt, metrics=['acc'])
model.summary()

lr_scheduler = LearningRateScheduler(lambda e: lr_start * lr_decay ** e)
history = model.fit(X_train, Y_train,
                    batch_size=batch_size, epochs=epochs,
                    verbose=1, validation_data=(X_test, Y_test),
                    callbacks=[lr_scheduler])

# Save model and weights
# if not os.path.isdir(save_dir):
#     os.makedirs(save_dir)
# model_path = os.path.join(save_dir, model_name)
model.save('./saved_models/grimaceModelMBDRelu.h5')
# print('Saved trained model at %s ' % model_path)

score = model.evaluate(X_test, Y_test, verbose=0)
# print(type(score))
# y_prob = model.predict(X_train) 
# y_classes = y_prob.argmax(axis=-1)
# print(y_prob)
# print(y_classes)
print('Test score:', score[0])
print('Test accuracy:', score[1])
