from keras import backend as K 
import tensorflow as tf 

def importancePenalty(weights, vs, norm=2):

	penalty = 0.0
	for weight in weights:
		penalty += tf.reduce_sum(vs['omega'][weight] * (weight - vs['cweights'][weight])**norm)
	return penalty

