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
# from helpers import protocols
# from helpers.keras_utils import LossHistory
# from helpers.optimizers import KOOptimizer

tf.logging.set_verbosity(tf.logging.INFO)

##EXTRACT LABELS
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

##CONSTRUCT DATASETS

# original train
train_original = np.zeros((60000,28,28), dtype=np.float32)
images_original = ["MNIST-processed-training/original/original{0}.png".format(k) for k in range(1,60000)]
for i in range(len(images_original)):
    img = np.array(Image.open(images_original[i]))
    train_original[i, :, :] = img
print(np.shape(train_original))

# original test
eval_original = np.zeros((10000,28,28), dtype=np.float32)
images2_original = ["MNIST-processed-test/original/test-original{0}.png".format(k) for k in range(1,10001)]
for i in range(len(images2_original)):
    img = np.array(Image.open(images2_original[i]))
    eval_original[i, :, :] = img
print(np.shape(eval_original))


# ROTATE 90 train
train_rot90 = np.zeros((60000,28,28), dtype=np.float32)
images_rot90 = ["MNIST-processed-training/rot90/rot90{0}.png".format(k) for k in range(1,60000)]
for i in range(len(images_rot90)):
    img = np.array(Image.open(images_rot90[i]))
    train_rot90[i, :, :] = img
print(np.shape(train_rot90))

# ROTATE 90 test
eval_rot90 = np.zeros((10000,28,28), dtype=np.float32)
images2_rot90 = ["MNIST-processed-test/rot90/test-rot90{0}.png".format(k) for k in range(1,10001)]
for i in range(len(images2_rot90)):
    img = np.array(Image.open(images2_rot90[i]))
    eval_rot90[i, :, :] = img
print(np.shape(eval_rot90))


# checkerboard train
train_checkerboard = np.zeros((60000,28,28), dtype=np.float32)
images_checkerboard = ["MNIST-processed-training/checkerboard/fullcheck{0}.png".format(k) for k in range(1,60000)]
for i in range(len(images_checkerboard)):
    img = np.array(Image.open(images_checkerboard[i]))
    train_checkerboard[i, :, :] = img
print(np.shape(train_checkerboard))

# checkerboard test
eval_checkerboard = np.zeros((10000,28,28), dtype=np.float32)
images2_checkerboard = ["MNIST-processed-test/checkerboard/test-checkerboard{0}.png".format(k) for k in range(1,10001)]
for i in range(len(images2_checkerboard)):
    img = np.array(Image.open(images2_checkerboard[i]))
    eval_checkerboard[i, :, :] = img
print(np.shape(eval_checkerboard))


# INV train
train_inv = np.zeros((60000,28,28), dtype=np.float32)
images_inv = ["MNIST-processed-training/Inv/inv{0}.png".format(k) for k in range(1,60000)]
for i in range(len(images_inv)):
    img = np.array(Image.open(images_inv[i]))
    train_inv[i, :, :] = img
print(np.shape(train_inv))

# INV test
eval_inv = np.zeros((10000,28,28), dtype=np.float32)
images2_inv = ["MNIST-processed-test/inv/test-inv{0}.png".format(k) for k in range(1,10001)]
for i in range(len(images2_inv)):
    img = np.array(Image.open(images2_inv[i]))
    eval_inv[i, :, :] = img
print(np.shape(eval_inv))


# fliplr train
train_fliplr = np.zeros((60000,28,28), dtype=np.float32)
images_fliplr = ["MNIST-processed-training/fliplr/fliplr{0}.png".format(k) for k in range(1,60000)]
for i in range(len(images_fliplr)):
    img = np.array(Image.open(images_fliplr[i]))
    train_fliplr[i, :, :] = img
print(np.shape(train_fliplr))

# fliplr test
eval_fliplr = np.zeros((10000,28,28), dtype=np.float32)
images2_fliplr = ["MNIST-processed-test/fliplr/test-fliplr{0}.png".format(k) for k in range(1,10001)]
for i in range(len(images2_fliplr)):
    img = np.array(Image.open(images2_fliplr[i]))
    eval_fliplr[i, :, :] = img
print(np.shape(eval_fliplr))


# flipud train
train_flipud = np.zeros((60000,28,28), dtype=np.float32)
images_flipud = ["MNIST-processed-training/flipud/flipud{0}.png".format(k) for k in range(1,60000)]
for i in range(len(images_flipud)):
    img = np.array(Image.open(images_flipud[i]))
    train_flipud[i, :, :] = img
print(np.shape(train_flipud)

# flipud test
eval_flipud = np.zeros((10000,28,28), dtype=np.float32)
images2_flipud = ["MNIST-processed-test/flipud/test-flipud{0}.png".format(k) for k in range(1,10001)]
for i in range(len(images2_flipud)):
    img = np.array(Image.open(images2_flipud[i]))
    eval_flipud[i, :, :] = img
print(np.shape(eval_flipud))


# cutud train
train_cutud = np.zeros((60000,28,28), dtype=np.float32)
images_cutud = ["MNIST-processed-training/cutud/cutUD{0}.png".format(k) for k in range(1,60000)]
for i in range(len(images_cutud)):
    img = np.array(Image.open(images_cutud[i]))
    train_cutud[i, :, :] = img
print(np.shape(train_cutud))

# cutud test
eval_cutud = np.zeros((10000,28,28), dtype=np.float32)
images2_cutud = ["MNIST-processed-test/cutud/test-cutud{0}.png".format(k) for k in range(1,10001)]
for i in range(len(images2_cutud)):
    img = np.array(Image.open(images2_cutud[i]))
    eval_cutud[i, :, :] = img
print(np.shape(eval_cutud))

# invbot train
train_invbot = np.zeros((60000,28,28), dtype=np.float32)
images_invbot = ["MNIST-processed-training/invbot/invbot{0}.png".format(k) for k in range(1,60000)]
for i in range(len(images_invbot)):
    img = np.array(Image.open(images_invbot[i]))
    train_invbot[i, :, :] = img
print(np.shape(train_invbot))

# invbot test
eval_invbot = np.zeros((10000,28,28), dtype=np.float32)
images2_invbot = ["MNIST-processed-test/invbot/test-invbot{0}.png".format(k) for k in range(1,10001)]
for i in range(len(images2_invbot)):
    img = np.array(Image.open(images2_invbot[i]))
    eval_invbot[i, :, :] = img
print(np.shape(eval_invbot))

##PARAMETERS
# Data params
img_rows, img_cols = 28, 28
# Optimization params
batch_size = 256
num_classes = 10
epochs = 5 # epochs per task


##DATA 
# label data
y_train = train_labels
y_test = eval_labels
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# 8 different dataset modifications
x_train = train_original
x_test = eval_original
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

x_train3 = train_inv
x_test3 = eval_inv
if K.image_data_format() == 'channels_first':
    x_train3 = x_train3.reshape(x_train3.shape[0], 1, img_rows, img_cols)
    x_test3 = x_test3.reshape(x_test3.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train3 = x_train3.reshape(x_train3.shape[0], img_rows, img_cols, 1)
    x_test3 = x_test3.reshape(x_test3.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
x_train3 = x_train3.astype('float32')
x_test3 = x_test3.astype('float32')
x_train3 /= 255
x_test3 /= 255

x_train4 = train_flipud
x_test4 = eval_flipud
if K.image_data_format() == 'channels_first':
    x_train4 = x_train4.reshape(x_train4.shape[0], 1, img_rows, img_cols)
    x_test4 = x_test4.reshape(x_test4.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train4 = x_train4.reshape(x_train4.shape[0], img_rows, img_cols, 1)
    x_test4 = x_test4.reshape(x_test4.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
x_train4 = x_train4.astype('float32')
x_test4 = x_test4.astype('float32')
x_train4 /= 255
x_test4 /= 255

x_train5 = train_fliplr
x_test5 = eval_fliplr
if K.image_data_format() == 'channels_first':
    x_train5 = x_train5.reshape(x_train5.shape[0], 1, img_rows, img_cols)
    x_test5 = x_test5.reshape(x_test5.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train5 = x_train5.reshape(x_train5.shape[0], img_rows, img_cols, 1)
    x_test5 = x_test5.reshape(x_test5.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
x_train5 = x_train5.astype('float32')
x_test5 = x_test5.astype('float32')
x_train5 /= 255
x_test5 /= 255

x_train6 = train_cutud
x_test6 = eval_cutud
if K.image_data_format() == 'channels_first':
    x_train6 = x_train6.reshape(x_train6.shape[0], 1, img_rows, img_cols)
    x_test6 = x_test6.reshape(x_test6.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train6 = x_train6.reshape(x_train6.shape[0], img_rows, img_cols, 1)
    x_test6 = x_test6.reshape(x_test6.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
x_train6 = x_train6.astype('float32')
x_test6 = x_test6.astype('float32')
x_train6 /= 255
x_test6 /= 255

x_train7 = train_invbot
x_test7 = eval_invbot
if K.image_data_format() == 'channels_first':
    x_train7 = x_train7.reshape(x_train7.shape[0], 1, img_rows, img_cols)
    x_test7 = x_test7.reshape(x_test7.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train7 = x_train7.reshape(x_train7.shape[0], img_rows, img_cols, 1)
    x_test7 = x_test7.reshape(x_test7.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
x_train7 = x_train7.astype('float32')
x_test7 = x_test7.astype('float32')
x_train7 /= 255
x_test7 /= 255

x_train8 = train_checkerboard
x_test8 = eval_checkerboard
if K.image_data_format() == 'channels_first':
    x_train8 = x_train8.reshape(x_train8.shape[0], 1, img_rows, img_cols)
    x_test8 = x_test8.reshape(x_test8.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train8 = x_train8.reshape(x_train8.shape[0], img_rows, img_cols, 1)
    x_test8 = x_test8.reshape(x_test8.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
x_train8 = x_train8.astype('float32')
x_test8 = x_test8.astype('float32')
x_train8 /= 255
x_test8 /= 255



##MODEL 1
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

##TRAIN 1
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Save trained model 1
model.save('trained_model.h5')
model2 = load_model('trained_model.h5')

##MODEL 2
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

# Save trained model 2
model2.save('trained_model2.h5')
model3 = load_model('trained_model2.h5')

##MODEL 3
# Continue training
model3.fit(x_train3, y_train, 
          batch_size=batch_size, 
          epochs=epochs, 
          verbose=1,
          validation_data=(x_test, y_test))
score1 = model3.evaluate(x_test, y_test, verbose=0)
score2 = model3.evaluate(x_test2, y_test, verbose=0)
score2 = model3.evaluate(x_test3, y_test, verbose=0)

print('Original data set')
print('Test loss:', score1[0])
print('Test accuracy:', score1[1])

print('Second data set')
print('Test loss:', score2[0])
print('Test accuracy:', score2[1])

print('Third data set')
print('Test loss:', score3[0])
print('Test accuracy:', score3[1])
