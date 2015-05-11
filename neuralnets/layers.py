import numpy as np

class Layer(object):
	""" Abstract class of a layer. """

	def __init__(self, input_shape):
		self.input_shape = input_shape
		self.output_shape = input_shape # Most layers don't change the shape
		self.acts = None
		self.gradient = None

	def forward(self, inputs, keepacts=False):
		assert inputs[0].shape == self.input_shape, 'Wrong input shape'
		return self._forward(inputs, keepacts)

	def backward(self, gradient, keepgrad=True):
		assert self.acts != [], "No activation values stored, backprop not possible"
		return self._backward(gradient, keepgrad)

	def update_parameters(self, learn_rate, regu_strength):
		""" To be implemented in child class (if needed). """
		pass

	def _forward(self, inputs, keepacts):
		""" To be implemented in child class. """
		raise NotImplementedError

	def _backward(self, gradient, keepgrad):
		""" To be implemented in child class. """
		raise NotImplementedError


class ConvLayer(Layer):
	# TODO: Implement the ConvLayer

	def __init__(self):
		# Hyperparameters
		self.n_filters = None
		self.field = None
		self.zero_pad = None
		self.stride = None
		 # Parameters = None
		self.weights = None
		self.biases = None

	def _forward(self, inputs, keepacts=False):
		pass

	def _backward(self, gradient, keepgrad=True):
		pass

	def update_parameters(self):
		pass


class PoolLayer(Layer):
	# TODO: Implement the PoolLayer

	def __init__(self):
		pass

	def _forward(self, inputs, keepacts=False):
		pass

	def _backward(self, gradient, keepgrad=True):
		pass

	def update_parameters():
		pass


class ReLuLayer(Layer):

	def _forward(self, inputs, keepacts=False):
		acts = np.maximum(0, inputs)
		if keepacts:
			self.acts = acts
		return acts

	def _backward(self, gradient, keepgrad=True):
		grad = gradient * (self.acts > 0)
		if keepgrad:
			self.gradient = grad
		return grad


class BiasLayer(Layer):

	def __init__(self, input_shape):
		super(BiasLayer, self).__init__(input_shape)
		self.biases = np.zeros(input_shape)

	def _forward(self, inputs, keepacts=False):
		acts = inputs + self.biases
		if keepacts:
			self.acts = acts
		return acts

	def _backward(self, gradient, keepgrad=True):
		if keepgrad:
			self.gradient = gradient
		return gradient

	def update_parameters(self, learn_rate, regu_strength):
		self.biases -= learn_rate * np.mean(self.gradient, axis=0)


class FCLayer(Layer):

	def __init__(self, input_shape, n_neurons):
		super(FCLayer, self).__init__(input_shape)
		n_input = np.prod(input_shape)
		self.weights = np.random.randn(n_input, n_neurons)
		self.weights *= np.sqrt(2.0 / n_input)
		self.output_shape = (n_neurons,)

	def _forward(self, inputs, keepacts=False):
		self.inputs = inputs
		acts = np.dot(inputs, self.weights)
		if keepacts:
			self.acts = acts
		return acts

	def _backward(self, gradient, keepgrad=True):
		grad = np.dot(self.inputs.T, gradient)
		if keepgrad:
			self.gradient = grad
		return np.dot(gradient, self.weights.T)

	def update_parameters(self, learn_rate, regu_strength):
		self.weights *= (1 - regu_strength)
		self.weights -= self.gradient * learn_rate / len(self.gradient)

