import tensorflow as tf
import pickle
from keras.utils import np_utils
from keras import backend as K
import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D, Layer

def set_gpu_config(device = "0",fraction=0.25):
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = fraction
    config.gpu_options.visible_device_list = device
    K.set_session(tf.Session(config=config))


class ODEBlock(Layer):

    def __init__(self, filters, kernel_size, **kwargs):
        self.filters = filters
        self.kernel_size = kernel_size
        super(ODEBlock, self).__init__(**kwargs)

    def build(self, input_shape):
        self.conv2d_w1 = self.add_weight("conv2d_w1", self.kernel_size + (self.filters + 1, self.filters), initializer='glorot_uniform')
        self.conv2d_w2 = self.add_weight("conv2d_w2", self.kernel_size + (self.filters + 1, self.filters), initializer='glorot_uniform')
        self.conv2d_b1 = self.add_weight("conv2d_b1", (self.filters,), initializer='zero')
        self.conv2d_b2 = self.add_weight("conv2d_b2", (self.filters,), initializer='zero')
        super(ODEBlock, self).build(input_shape)

    def call(self, x):
        t = K.constant([0, 1], dtype="float32")
        return tf.contrib.integrate.odeint(self.ode_func, x, t, rtol=1e-3, atol=1e-3)[1]

    def compute_output_shape(self, input_shape):
        return input_shape

    def ode_func(self, x, t):
        y = self.concat_t(x, t)
        y = K.conv2d(y, self.conv2d_w1, padding="same")
        y = K.bias_add(y, self.conv2d_b1)
        y = K.relu(y)

        y = self.concat_t(y, t)
        y = K.conv2d(y, self.conv2d_w2, padding="same")
        y = K.bias_add(y, self.conv2d_b2)
        y = K.relu(y)

        return y

    def concat_t(self, x, t):
        new_shape = tf.concat(
            [
                tf.shape(x)[:-1],
                tf.constant([1],dtype="int32",shape=(1,))
            ], axis=0)

        t = tf.ones(shape=new_shape) * tf.reshape(t, (1, 1, 1, 1))
        return tf.concat([x, t], axis=-1)


def build_model(input_shape, num_classes):
    x = Input(input_shape)
    y = Conv2D(32, (3, 3),padding='same', use_bias=use_bias, activation='relu')(x)
    y = MaxPooling2D((1,1))(y)
    y = Conv2D(64, (3, 3), activation='relu')(y)
    y = MaxPooling2D((2,2))(y)
    y = ODEBlock(64, (3, 3))(y)
    y = Flatten()(y)
    y = Dense(num_classes, activation='softmax')(y)
    return Model(x,y)


set_gpu_config("0",0.25)

# batch_size = 128
# num_classes = 10
# epochs = 10
# image_shape = (28, 28, 1)

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train = x_train.astype('float32') / 255.0
# x_test = x_test.astype('float32') / 255.0

# x_train = x_train.reshape((-1,) + image_shape)
# x_test = x_test.reshape((-1,) + image_shape)

# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test  = keras.utils.to_categorical(y_test, num_classes)


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
num_classes = 72
use_bias = False
image_shape = (71,64,1)

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

X_train = X_train.reshape(1224, 71,64,1)
X_test = X_test.reshape(216, 71,64,1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes) * 2 - 1 # -1 or 1 for hinge loss
Y_test = np_utils.to_categorical(y_test, nb_classes) * 2 - 1







model = build_model(image_shape, num_classes)

model.compile(loss=keras.losses.categorical_hinge,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

model.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, Y_test))

model.save('./saved_models/OdeNetF95.h5')

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
