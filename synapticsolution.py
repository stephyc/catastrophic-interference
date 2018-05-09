from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
import numpy as np
from keras.utils import np_utils
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
from utils import ema
from skdata.larochelle_etal_2007 import dataset as l7 
import fisher_comp

from helpers import utils 
from synapticpenalty import importancePenalty
from optimizers import SynapticOptimizer as SO 

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
        #('cached_grads2', lambda vars, w, prev_val: vars['grads2'][w]),
        #('cached_cweights', lambda vars, w, prev_val: vars['cweights'][w]),
        ('cweights',  lambda opt, w, prev_val: w.value()),
        ('grads2', lambda vars, w, prev_val: prev_val*0.0 ),
    ],
    'regularizer_fn': importancePenalty,
})

# Optimization parameters
img_rows, img_cols = 28, 28
train_size = 50000
valid_size = 10000
test_size = 10000
batch_size = 256
num_classes = 10
epochs = 10
lr = 0.001
xi = 0.1

# Architecture params
hidden_neurons = 2000
activation_fn = tf.nn.relu 
output_fn = tf.nn.softmax

# data
input_size = 784
output_size = 10

# reset the weight state of a model
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
valid_labels = train_labels[50000:]
train_labels = train_labels[:50000]
eval_labels = extract_labels("MNIST-data/t10k-labels-idx1-ubyte.gz", 10000)
y_train = keras.utils.to_categorical(train_labels, num_classes)
y_valid = keras.utils.to_categorical(valid_labels, num_classes)
y_test = keras.utils.to_categorical(eval_labels, num_classes)

# create local dataset based off of permuted MNIST data
def createDataset(name, trainsrc, testsrc):
	print("Beginning import:", name)
	train = np.zeros((train_size, img_rows, img_cols), dtype=np.float32)
	test = np.zeros((test_size, img_rows, img_cols), dtype=np.float32)
	valid = np.zeros((valid_size, img_rows, img_cols), dtype=np.float32)

	# create and import image sets
	imgstrain = ["{0}{1}.png".format(trainsrc, k) for k in range(1, train_size)]
	imgstest = ["MNIST-processed-test/{0}{1}.png".format(testsrc, k) for k in range(1, test_size + 1)]
	imgsvalid = imgstrain[50000:]
	imgstrain = imgstrain[:50000]

	# reshape data
	for i in range(len(imgstrain)):
		img = np.array(Image.open(imgstrain[i]))
		train[i, :, :] = img

	for i in range(len(imgsvalid)):
		img = np.array(Image.open(imgsvalid[i]))
		valid[i, :, :] = img

	for  i in range(len(imgstest)):
		img = np.array(Image.open(imgstest[i]))
		test[i, :, :] = img

	# tracking message
	print("Completed import:", name)

	return (train, test, valid)

reset_optimizer = False
# import skdata sets
def createRandomizedDataset():

	# chosen tasks
	tasks = [l7.MNIST_Basic(), l7.MNIST_Rotated(), l7.MNIST_Noise1(), \
			 l7.MNIST_Noise3(), l7.MNIST_Noise5()]

	# costruct datasets
	datasets = dict()
	for i in range(len(tasks)):
		name = tasks[i]
		task = name.classification_task()
		raw_data, raw_labels = task 
		classes = 10
		
		labels = np_utils.to_categorical(raw_labels, classes)

		data = raw_data.reshape(raw_data.shape[0], img_rows * img_cols)
		
		data = raw_data
		training_ex = int(len(data) * 5/7)
		valid_ex = int(len(data) * 1/7) + training_ex

		datasets[i] = {"train": data[:training_ex], "test": data[valid_ex:], \
						"valid": data[training_ex : valid_ex], "validlabels": labels[training_ex : valid_ex], \
		               "trainlabels": labels[:training_ex], "tstlabels": labels[valid_ex:]}
	return datasets

# import local filepath data
def importData():
	srcs = [("original", "original/original/original", "original/original/test-original"),\
			("rot90", "rot90/rot90", "rot90/rot90/test-rot90"), \
			("fliplr", "fliplr/fliplr/fliplr", "fliplr/fliplr/test-fliplr"), \
			("flipud", "flipud/flipud/flipud", "flipud/flipud/test-flipud"), \
			("check", "checkerboard/checkerboard/fullcheck", "checkerboard/checkerboard/test-checkerboard"), \
			#("inv", "Inv/Inv/inv", "inv/inv/test-inv"), \
			#("cutud","cutud/cutud/cutUD", "cutud/cutud/test-cutud"),\
			#("invbot", "invbot/invbot/invbot", "invbot/invbot/test-invbot"), ]
			]
	# construct datasets
	datasets = list(map(lambda x: createDataset(x[0], x[1], x[2]), srcs))

	data = dict()


	for i in range(len(datasets)):
		x_train = datasets[i][0]
		x_test = datasets[i][1]
		x_valid = datasets[i][2]
		#if K.image_data_format() == 'channels_first':
		x_train = x_train.reshape(x_train.shape[0], img_rows * img_cols)
		x_test = x_test.reshape(x_test.shape[0], img_rows * img_cols)
		x_valid = x_valid.reshape(x_valid.shape[0], img_rows * img_cols)
		input_shape = (1, img_rows * img_cols)

		"""
		else:
		    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
		    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
		    input_shape = (img_rows, img_cols, 1)
		"""

		x_train = x_train.astype('float32')
		x_test = x_test.astype('float32')
		x_valid = x_valid.astype('float32')
		x_train /= 255
		x_test /= 255
		x_valid /= 255

		data[i] = {"train": x_train, "test": x_test, "valid": x_val}

	return data

# open file for output data
fn = "solutionresults"


# build model
model = Sequential()
model.add(Dense(hidden_neurons, activation=activation_fn, input_dim=input_size))
model.add(Dense(hidden_neurons, activation=activation_fn))
model.add(Dense(output_size, activation=output_fn))

# build optimizer
opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
pro_name, pro = PATH_INT_PROTOCOL(omega_decay="sum", xi=xi)
oopt = SO(opt=opt, model=model, **pro)

# create output data files
oopt.createFiles("fishers.txt", "weights.txt")

# compile model
model.compile(loss=keras.losses.categorical_crossentropy,
			  optimizer=oopt,
			  metrics=['accuracy'])

# create dataset (either skdata or customized sets)
data = createRandomizedDataset()
config = tf.ConfigProto()
#data = importData()
sess = tf.InteractiveSession(config=config)
sess.run(tf.global_variables_initializer())

ntasks = len(data)

# decide training order (or randomize order)
training_order = [2, 1, 0, 3, 4]
#training_order = training_order[0:5]

# order of strengths to regularize
strengths = [0.0, 0.2, 0.5, 1.0]

evals = dict()

# train network on strengths
for strength in strengths:
	print("setting strength: {0}".format(strength))
	oopt.set_strength(strength)
	evals[strength] = dict()

	# train network on chosen task
	for train in training_order:
		filename = "{0}{1}random2retrial{2}.txt".format(fn, train, strength)
		file = open(filename, 'w+')
		evals[strength][train] = list()
		mess = "Training on task {0}".format(train)
		print(mess)
		file.write(mess)
		oopt.set_nb_data(len(data[train]["train"]))
		model.fit(data[train]["train"], data[train]["trainlabels"],
				  batch_size=batch_size,
				  epochs=epochs,
				  verbose=1,
				  validation_data=(data[train]["valid"], data[train]["validlabels"]))

		# test on other tasks
		scores = dict()
		for d in range(len(data.keys())):
			scores[d] = model.evaluate(data[d]["test"], data[d]["tstlabels"], verbose=0)
			mess = "Data set {0}:\nScore:{1}\n".format(d, scores[d])
			print(mess)
			file.write(mess)
		file.close() 

		# print weight image, weights
		oopt.outputImageData(train, strength)
		oopt.print_weight_state()
		oopt.print_fisher_state()

	# potentially reset optimizer
	if reset_optimizer:
		 oopt.reset_optimizer()



# close open files
oopt.closeFiles()







