import numpy as np
import scipy.signal

# TODO: Add PoolLayer
# TODO: Add DropoutLayer
# TODO: Add a more activation functions

class Layer(object):
	""" Abstract class of a layer. """

	# TODO: Add a gradient checking method

	def __init__(self, input_shape):
		self.input_shape = input_shape
		self.output_shape = input_shape # Most layers don't change the shape
		self.acts = None
		self.velocity = None # Must be set in subclass for params update

	def forward(self, inputs, keepacts=False):
		""" Forwards the activations through this layer.

		inputs: the inputs of the layer. Must be of the same shape as given
		at intialization.
		keepacts: True if backpropagation is planned (for training)

		This method is a wrapper around _forward() that must be implemented
		in the subclasses.
		"""
		assert inputs.shape[1:] == self.input_shape[1:], 'Wrong input shape'
		acts = self._forward(inputs, keepacts)
		if keepacts:
			self.acts = acts
		else:
			self.acts = None # Freeing memory, the activations can be heavy
		return acts

	def backward(self, gradient):
		""" Propagates the given output gradient through the layer.

		gradient: the error gradient on the output of the layer.

		This method is a wrapper around _backward() that must be implemented
		in the subclasses.
		"""
		assert self.acts is not None, ("No activation values stored, "
				"backprop not possible")
		return self._backward(gradient)

	def update_parameters(self, learn_rate, momentum):
		""" To be implemented in child class (if needed). """
		pass

	def _forward(self, inputs, keepacts):
		""" Must implemented in child class. """
		raise NotImplementedError

	def _backward(self, gradient):
		""" Must implemented in child class. """
		raise NotImplementedError


class ConvLayer(Layer):
	""" A convolutional layer.

	Note that it doesn't apply a bias nor an activation function.
	"""

	def __init__(self, input_shape,	n_filters, field=3):
		""" Creates a convolution layer with n filters.

		n_filters: the number of filters (kernels) contained in this layer
		field: the length of the side of a (square) filter

		input_shape is expected to be (num_data, num_chan, x_rez, y_rez).
		"""
		super(ConvLayer, self).__init__(input_shape)
		# Hyperparameters
		self.n_filters = n_filters
		self.field = field
		 # Parameters
		self.weights = np.random.randn(n_filters, input_shape[1],\
				field, field)
		for i in xrange(n_filters):
			self.weights[i] *= np.sqrt(2.0 / np.sum(self.weights[i].shape))
		self.d_weights = None
		self.velocity = np.zeros(self.weights.shape)
		self.output_shape = (1, self.n_filters,\
				input_shape[2], input_shape[3])

	def _forward(self, inputs, keepacts=False):
		""" Uses scipy to convolve each channel of each image separately
			which is painfully slow. """
		self.inputs =  inputs
		outputs = np.zeros((inputs.shape[0],) + self.output_shape[1:])
		for n in xrange(inputs.shape[0]): # for each image
			for k in xrange(self.n_filters): # for each kernel
				for c in xrange(inputs.shape[1]): # for each channel

					outputs[n][k] += scipy.signal.convolve2d(inputs[n][c],\
							self.weights[k][c], mode='same')
		return outputs


	def _backward(self, gradient):
		""" The slowest backprop you will ever see for a convolutional
			layer. """
		self.d_weights = np.zeros(self.weights.shape)
		grad = np.zeros(self.inputs.shape)
		for k in xrange(self.n_filters): # for each kernel
			for c in xrange(self.inputs.shape[1]): # for each channel
				for n in xrange(self.inputs.shape[0]): # for each image

					self.d_weights[k][c] += scipy.signal.convolve2d(\
							self.inputs[n][c], gradient[n][k], 'valid')

					grad[n][c] += scipy.signal.correlate2d(gradient[n][k],\
							self.weights[k][c], 'same')
		return grad

	def update_parameters(self, learn_rate, momentum):
		self.velocity = momentum * self.velocity - self.d_weights * learn_rate
		self.weights += self.velocity


class ReLuLayer(Layer):
	""" An activation layer that uses the ReLu function. (max(0, input)) """

	def _forward(self, inputs, keepacts=False):
		acts = np.maximum(0, inputs)
		return acts

	def _backward(self, gradient):
		grad = gradient * (self.acts > 0)
		return grad


class BiasLayer(Layer):

	def __init__(self, input_shape):
		super(BiasLayer, self).__init__(input_shape)
		self.biases = np.zeros(input_shape[1:])
		self.d_biases = None
		self.velocity = np.zeros(input_shape[1:])

	def _forward(self, inputs, keepacts=False):
		acts = inputs + self.biases
		return acts

	def _backward(self, gradient):
		self.d_biases = gradient
		return gradient

	def update_parameters(self, learn_rate, momentum):
		self.velocity = momentum * self.velocity
		self.velocity -= learn_rate * np.mean(self.d_biases, axis=0)
		self.biases -= self.velocity


class FCLayer(Layer):
	""" Fully Connected Layer.

	All the neurons are connected to every neuron in the previous layer.
	"""

	def __init__(self, input_shape, n_neurons):
		"""
		input_shape: a tuple expected to be (N, C, X, X)
		n_neurons: the number of neurons in this layer.
		"""
		super(FCLayer, self).__init__(input_shape)
		n_input = np.prod(input_shape[1:])
		self.weights = np.random.randn(n_input, n_neurons)
		self.weights *= np.sqrt(2.0 / n_input)
		self.output_shape = (input_shape, n_neurons)
		self.d_weights = None
		self.velocity = np.zeros(self.weights.shape)

	def _forward(self, inputs, keepacts=False):
		inputs = inputs.reshape(inputs.shape[0], np.prod(self.input_shape[1:]))
		self.inputs = inputs # FIXME: This shouldn't be there as such
		acts = np.dot(inputs, self.weights)
		return acts

	def _backward(self, gradient):
		self.d_weights = np.dot(self.inputs.T, gradient)
		grad = np.dot(gradient, self.weights.T)
		return grad.reshape((gradient.shape[0],) + self.input_shape[1:])

	def update_parameters(self, learn_rate, momentum):
		self.velocity = momentum * self.velocity - self.d_weights * learn_rate
		self.weights += self.velocity

