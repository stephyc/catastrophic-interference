""" keras optimizer implementing synaptic solution """

import tensorflow as tf 
import numpy as np 
import keras
from keras import backend as K 
from keras.optimizers import Optimizer 
#from keras.callbacks import Callback 
from synapticpenalty import importancePenalty
from collections import OrderedDict 

class SynapticOptimizer(Optimizer):

		def _allocate_var(self, name=None):
				return {w: K.zeros(w.get_shape(), name=name) for w in self.weights}

		def _allocate_vars(self, names):
				self.vars = {name:self._allocate_var(name=name) for name in names}

		def __init__(self, opt, step_updates=[], task_updates=[], init_updates=[], task_metrics = {}, regularizer_fn=importancePenalty,
	    lam=1.0, model=None, compute_average_loss=False, compute_average_weights=False, **kwargs):
				
			super(SynapticOptimizer, self).__init__(**kwargs)

			if not isinstance(opt, keras.optimizers.Optimizer):
					raise ValueError("Optimizer must be instance of keras.optimizers.Optimizer")

			self.regularizer_fn = regularizer_fn
			self.task_updates = OrderedDict(task_updates)
			self.step_updates = OrderedDict(step_updates)
			self.init_updates = OrderedDict(init_updates)
			self.task_metrics = task_metrics

			self.names = set().union(self.step_updates.keys(), self.task_updates.keys(), self.task_metrics.keys())
			self.lam = K.variable(lam, dtype=tf.float32, name="lam")
			self.nb_data = K.variable(value=1.0, dtype=tf.float32, name="nb_data")
			self.model = model
			self.opt = opt
			self.compute_average_weights = compute_average_weights

 
		def set_strength(self, val):
				K.set_value(self.lam, val)

		def set_nb_data(self, nb):
				K.set_value(self.nb_data, nb)

		def get_updates(self, loss, params):
				

				self.weights = self.get_weights()

				with tf.variable_scope("SynapticOptimizer"):
						self._allocate_vars(self.names)

				self.regularizer = self.regularizer_fn(self.weights, self.vars)
				self.initial_loss = loss
				self.loss = loss + self.lam * self.regularizer

				with tf.variable_scope("wrapped_optimizer"):
						self._weight_update_op, self._grads, self._deltas = self.compute_updates(self.opt, self.loss, params)

				wrapped_opt_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "wrapped_optimizer")
				self.init_opt_vars = tf.variables_initializer(wrapped_opt_vars)

				self.vars['unreg_grads'] = dict(zip(self.weights, tf.gradients(self.initial_loss, self.weights)))
        		# Compute updates
				self.vars['grads'] = dict(zip(self.weights, self._grads))
				self.vars['deltas'] = dict(zip(self.weights, self._deltas))
				# Keep a pointer to self in vars so we can use it in the updates
				self.vars['oopt'] = self
				# Keep number of data samples handy for normalization purposes
				self.vars['nb_data'] = self.nb_data

				def _var_update(vs, update_fn):
					updates = []
					for w in self.weights:
							updates.append(tf.assign(vars[w], update_fn(self.vars, w, vs[w])))
					return tf.group(*updates)

				def _compute_vars_update_op(updates):
	            # Force task updates to happen sequentially
					update_op = tf.no_op()
					for name, update_fn in updates.items():
							with tf.control_dependencies([update_op]):
									update_op = _var_update(self.vars[name], update_fn)
					return update_op

				self._vars_step_update_op = _compute_vars_update_op(self.step_updates)
				self._vars_task_update_op = _compute_vars_update_op(self.task_updates)
				self._vars_init_update_op = _compute_vars_update_op(self.init_updates)


				reset_ops = []
				update_ops = []
				for name, metric_fn in self.task_metrics.items():
					metric = metric_fn(self)
					for w in self.weights:
							reset_ops.append(tf.assign(self.vars[name][w], 0*self.vars[name][w]))
							update_ops.append(tf.assign_add(self.vars[name][w], metric[w]))
				self._reset_task_metrics_op = tf.group(*reset_ops)
				self._update_task_metrics_op = tf.group(*update_ops)

				# Each step we update the weights using the optimizer as well as the step-specific variables
				self.step_op = tf.group(self._weight_update_op, self._vars_step_update_op)
				self.updates.append(self.step_op)
			
				# After each task, run task-specific variable updates
				self.task_op = self._vars_task_update_op
				self.init_op = self._vars_init_update_op
				
				return self.updates


		def get_weights(self):
			return K.batch_get_value(self.weights)

		def compute_updates(self, opt, loss, params):
			update_ops = opt.get_updates(loss, params)
			deltas, new_update_op = self.extract_weight_changes(self.weights, update_ops)
			grads = tf.gradients(loss, self.weights)
			# Make sure  that deltas are computed _before_ the weight is updated
			return new_update_op, grads, deltas

		def extract_weight_changes(self, weights, update_ops):

			name_to_var = {v.name: v.value() for v in weights}
			weight_update_ops = list(filter(lambda x: x.op.inputs[0].name in name_to_var, update_ops))
			nonweight_update_ops = list(filter(lambda x: x.op.inputs[0].name not in name_to_var, update_ops))
    
			# Make sure that all the weight update ops are Assign ops
			for weight in weight_update_ops:
				if weight.op.type != 'Assign':
					raise ValueError('Update op for weight %s is not of type Assign.'%weight.op.inputs[0].name)
			weight_changes = [(new_w.op.inputs[1] - name_to_var[new_w.op.inputs[0].name]) for new_w, old_w in zip(weight_update_ops, weights)]
    
			# Recreate the update ops, ensuring that we compute the weight changes before updating the weights
			with tf.control_dependencies(weight_changes):
				new_weight_update_ops = [tf.assign(new_w.op.inputs[0], new_w.op.inputs[1]) for new_w in weight_update_ops]
			return weight_changes, tf.group(*(nonweight_update_ops + new_weight_update_ops))

			