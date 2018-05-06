from keras import backend as K 
import tensorflow as tf 

def importancePenalty(weights, vars, norm=2):

	penalty = 0.0
	for weight in weights:
		penalty += tf.reduce_sum(vars['omega'][weight] * (weight - vars['cweights'][weight])**norm)
	return penalty

