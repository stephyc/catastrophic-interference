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
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
import matplotlib


import keras
from keras import backend as K 
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop, Optimizer 
from keras.callbacks import Callback 
from collections import OrderedDict 

from helpers import utils 
from synapticpenalty import importancePenalty
from kerasoptimizer import SynapticOptimizer as SO 

tf.logging.set_verbosity(tf.logging.INFO)

# Pathint protocol describes a set of kwargs for the SynapticOptimizer (credit: Zenke et. al, 2017)
PATH_INT_PROTOCOL = lambda omega_decay, xi: (
        'path_int[omega_decay=%s,xi=%s]'%(omega_decay,xi),
{
    'init_updates':  [
        ('cweights', lambda vars, w, prev_val: w.value() ),
        ],
    'step_updates':  [
        ('grads2', lambda vars, w, prev_val: prev_val -vars['unreg_grads'][w] * vars['deltas'][w] ),
        ],
    'task_updates':  [
        ('omega',     lambda vars, w, prev_val: tf.nn.relu( ema(omega_decay, prev_val, vars['grads2'][w]/((vars['cweights'][w]-w.value())**2+xi)) ) ),
        ('cweights',  lambda opt, w, prev_val: w.value()),
        ('grads2', lambda vars, w, prev_val: prev_val*0.0 ),
    ],
    'regularizer_fn': importancePenalty,
})

# Optimization parameters
img_rows, img_cols = 28, 28
train_size = 60000
test_size = 10000
batch_size = 256
num_classes = 10
epochs = 5
lr = 0.01
xi = 0.1

# Architecture params
hidden_neurons = 3000
activation_fn = tf.nn.relu 
output_fn = tf.nn.softmax

# data
input_size = 784
output_size = 10

def reset_weights(model):
    session = K.get_session()
    for layer in model.layers:
        if isinstance(layer, Dense):
            old = layer.get_weights()
            layer.W.initializer.run(session=session)
            layer.b.initializer.run(session=session)
            print(np.array_equal(old, layer.get_weights())," after initializer run")
        else:
            print(layer, "not reinitialized")

def shuffleOrder(rnge):
	return np.random.permutation([i for i in range(rnge)])

# Get data labels from local file
def extract_labels(filename, num_images):
	with gzip.open(filename) as bytestream:
		bytestream.read(8)
		buf = bytestream.read(1 * num_images)
		labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
		return labels

# extracrt labels from local file
train_labels = extract_labels("MNIST-data/train-labels-idx1-ubyte.gz", 60000)
eval_labels = extract_labels("MNIST-data/t10k-labels-idx1-ubyte.gz", 10000)
y_train = keras.utils.to_categorical(train_labels, num_classes)
y_test = keras.utils.to_categorical(eval_labels, num_classes)

# create local dataset based off of permuted MNIST data
def createDataset(name, trainsrc, testsrc):
	print("Beginning import:", name)
	train = np.zeros((train_size, img_rows, img_cols), dtype=np.float32)
	test = np.zeros((test_size, img_rows, img_cols), dtype=np.float32)

	imgstrain = ["{0}{1}.png".format(trainsrc, k) for k in range(1, train_size)]
	imgstest = ["MNIST-processed-test/{0}{1}.png".format(testsrc, k) for k in range(1, test_size + 1)]

	for i in range(len(imgstrain)):
		img = np.array(Image.open(imgstrain[i]))
		train[i, :, :] = img

	for  i in range(len(imgstest)):
		img = np.array(Image.open(imgstest[i]))
		test[i, :, :] = img

	print("Completed import:", name)

	return (train, test)

# import all permuted datasets (paths = local paths in filesystem)
def importData():
	srcs = [("original", "original/original/original", "original/original/test-original"),\
			("rot90", "rot90/rot90", "rot90/rot90/test-rot90"), \
			("fliplr", "fliplr/fliplr/fliplr", "fliplr/fliplr/test-fliplr"), #\
			#("flipud", "flipud/flipud/flipud", "flipud/flipud/test-flipud"), \
			#("check", "checkerboard/checkerboard/fullcheck", "checkerboard/checkerboard/test-checkerboard"), \
			#("inv", "Inv/Inv/inv", "inv/inv/test-inv"), \
			#("cutud","cutud/cutud/cutUD", "cutud/cutud/test-cutud"),\
			#("invbot", "invbot/invbot/invbot", "invbot/invbot/test-invbot"), ]
			]

	datasets = list(map(lambda x: createDataset(x[0], x[1], x[2]), srcs))

	data = dict()


	for i in range(len(datasets)):
		x_train = datasets[i][0]
		x_test = datasets[i][1]
		#if K.image_data_format() == 'channels_first':
		x_train = x_train.reshape(x_train.shape[0], img_rows * img_cols)
		x_test = x_test.reshape(x_test.shape[0], img_rows * img_cols)
		input_shape = (1, img_rows * img_cols)

		"""
		else:
		    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
		    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
		    input_shape = (img_rows, img_cols, 1)
		"""

		x_train = x_train.astype('float32')
		x_test = x_test.astype('float32')
		x_train /= 255
		x_test /= 255

		data[i] = {"train": x_train, "test": x_test}

	return data

# open file for output data
file = open("solutionresults2.txt", 'w+')

# build model
model = Sequential()
model.add(Dense(hidden_neurons, activation=activation_fn, input_dim=input_size))
model.add(Dense(hidden_neurons, activation=activation_fn))
model.add(Dense(output_size, activation=output_fn))

# build optimizer
opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
pro_name, pro = PATH_INT_PROTOCOL(omega_decay="sum", xi=xi)

oopt = SO(opt=opt, model=model, **pro)

model.compile(loss=keras.losses.categorical_crossentropy,
			  optimizer=oopt,
			  metrics=['accuracy'])

data = importData()

ntasks = len(data)

training_order = shuffleOrder(len(data))
training_order = training_order[0:5]
strengths = [0.1, 1.0]

evals = dict()

for strength in strengths:
	print("setting strength: {0}".format(strength))
	oopt.set_strength(strength)
	evals[strength] = dict()

	for train in training_order:
		evals[strength][train] = list()
		mess = "Training on task {0}".format(train)
		print(mess)
		file.write(mess)
		oopt.set_nb_data(len(data[train]["train"]))
		model.fit(data[train]["train"], y_train,
				  batch_size=batch_size,
				  epochs=epochs,
				  verbose=1,
				  validation_data=(data[train]["test"], y_test))

		scores = dict()
		for d in range(len(data.keys())):
			scores[d] = model.evaluate(data[d]["test"], y_test, verbose=0)
			mess = "Data set {0}:\nTest loss:{1}\ntest accuracy:{2}\n".format(d, scores[d][0], scores[d][1])
			print(mess)
			file.write(mess)

	model = Sequential()
	model.add(Dense(hidden_neurons, activation=activation_fn, input_dim=input_size))
	model.add(Dense(hidden_neurons, activation=activation_fn))
	model.add(Dense(output_size, activation=output_fn))

	opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	pro_name, pro = PATH_INT_PROTOCOL(omega_decay="sum", xi=xi)

	oopt = SO(opt=opt, model=model, **pro)

	model.compile(loss=keras.losses.categorical_crossentropy,
			  optimizer=oopt,
			  metrics=['accuracy'])

file.close() 

"""
cNorm  = colors.Normalize(vmin=-5, vmax=np.log(np.max(list(evals.keys()))))
scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cmap)
print(scalarMap.get_clim())

plt.figure(figsize=(14, 4))
axs = [plt.subplot(1, ntasks + 1, 1)]
for i in range(1, ntasks + 1):
	axs.append(plt.subplot(1, ntasks + 1, i + 1, sharex=axs[0], sharey = axs[0]))

fmts = ['o', 's']
keys = sorted(evals.keys())

for strength in keys:
	results = evals[strength]
	label = "c=%g"%strength
	for i in range(ntasks + 1):
		col = scalarMap.to_rgba(np.log(strength))
		
		axs[i].plot(results[:, i], c=col, label=label)

for i, ax in enumerate(axs):
	ax.legend(bbox_to_anchor=(1.0, 1.0))
	ax.set_title((['task %d'%j for j in range(n_tasks)] + ['average'])[i])
gcf().tight_layout()
"""








