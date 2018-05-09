from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
import numpy as np
import scipy 
import tensorflow as tf

import imageio
import gzip
from PIL import Image

import seaborn as sns
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
#from tqdm import trange, tqdm

import keras
from keras import backend as K
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop, Optimizer
from keras.callbacks import Callback
from collections import OrderedDict

from helpers import utils

tf.logging.set_verbosity(tf.logging.INFO)

# Extract labels from MNIST labels into vector
def extract_labels(filename, num_images):
 with gzip.open(filename) as bytestream:
   bytestream.read(8)
   buf = bytestream.read(1 * num_images)
   labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
 return labels

train_labels = extract_labels("MNIST-data/train-labels-idx1-ubyte.gz", 60000)
eval_labels = extract_labels("MNIST-data/t10k-labels-idx1-ubyte.gz", 10000)
print(np.shape(train_labels))
print(np.shape(eval_labels))

# original train
train_original = np.zeros((60000,28,28), dtype=np.float32)
images_original = ["MNIST-processed-training/original/original{0}.png".format(k) for k in range(1,60000)]
for i in range(len(images_original)):
   img = np.array(Image.open(images_original[i]))
   train_original[i, :, :] = img

# original test
eval_original = np.zeros((10000,28,28), dtype=np.float32)
images2_original = ["MNIST-processed-test/original/test-original{0}.png".format(k) for k in range(1,10001)]

for i in range(len(images2_original)):
   img = np.array(Image.open(images2_original[i]))
   eval_original[i, :, :] = img

# ROTATE 90 train
train_rot90 = np.zeros((60000,28,28), dtype=np.float32)
images_rot90 = ["MNIST-processed-training/rot90/rot90{0}.png".format(k) for k in range(1,60000)]

for i in range(len(images_rot90)):
   img = np.array(Image.open(images_rot90[i]))
   train_rot90[i, :, :] = img

# ROTATE 90 test
eval_rot90 = np.zeros((10000,28,28), dtype=np.float32)
images2_rot90 = ["MNIST-processed-test/rot90/test-rot90{0}.png".format(k) for k in range(1,10001)]

for i in range(len(images2_rot90)):
   img = np.array(Image.open(images2_rot90[i]))
   eval_rot90[i, :, :] = img

# input image dimensions
img_rows, img_cols = 28, 28

# # Network params
# n_hidden_units = 2000
# activation_fn = tf.nn.relu

# Optimization params
batch_size = 256
num_classes = 10
epochs = 5 # epochs per task
# learning_rate=1e-3
# xi = 0.1

# the data, train and test sets
x_train = train_original
x_test = eval_original
y_train = train_labels
y_test = eval_labels


if K.image_data_format() == 'channels_first':
   x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
   x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
   input_shape = (1, img_rows, img_cols)
else:
   x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
   x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
   input_shape = (img_rows, img_cols, 1)
   
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                activation='relu',
                input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
             optimizer=keras.optimizers.Adadelta(),
             metrics=['accuracy'])


model.fit(x_train, y_train,
         batch_size=batch_size,
         epochs=epochs,
         verbose=1,
         validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])



model2 = Sequential()

# Convolutional layer 1 and input layer
model2.add(Conv2D(32, kernel_size=(3, 3),
                activation='relu',
                input_shape=input_shape))

# Convolutional layer 2
model2.add(Conv2D(64, (3, 3), activation='relu'))

# Pooling layer 1
model2.add(MaxPooling2D(pool_size=(2, 2)))

# Dropout layer with flattening
model2.add(Dropout(0.25))
model2.add(Flatten())

# Dense layer 1 with dropout 
model2.add(Dense(128, activation='relu'))
model2.add(Dropout(0.5))

# Dense layer 2
model2.add(Dense(num_classes, activation='softmax'))

model2.compile(loss=keras.losses.categorical_crossentropy,
             optimizer=keras.optimizers.Adadelta(),
             metrics=['accuracy'])

model.save('trained_model.h5')
model2 = load_model('trained_model.h5')

x_train2 = train_rot90
x_test2 = eval_rot90


if K.image_data_format() == 'channels_first':
   x_train2 = x_train2.reshape(x_train2.shape[0], 1, img_rows, img_cols)
   x_test2 = x_test2.reshape(x_test2.shape[0], 1, img_rows, img_cols)
   input_shape = (1, img_rows, img_cols)
else:
   x_train2 = x_train2.reshape(x_train2.shape[0], img_rows, img_cols, 1)
   x_test2 = x_test2.reshape(x_test2.shape[0], img_rows, img_cols, 1)
   input_shape = (img_rows, img_cols, 1)
   
x_train2 = x_train2.astype('float32')
x_test2 = x_test2.astype('float32')
x_train2 /= 255
x_test2 /= 255


# Continue training
model2.fit(x_train2, y_train,
         batch_size=batch_size,
         epochs=epochs,
         verbose=1,
         validation_data=(x_test, y_test))
score1 = model2.evaluate(x_test, y_test, verbose=0)
score2 = model2.evaluate(x_test2, y_test, verbose=0)

print('Original data set')
print('Test loss:', score1[0])
print('Test accuracy:', score1[1])

print('Second data set')
print('Test loss:', score2[0])
print('Test accuracy:', score2[1])