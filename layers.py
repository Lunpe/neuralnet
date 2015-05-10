import numpy as np

class Layer():
	""" Abstract class of a layer. """

	def __init__(self, shape):
		self.width = shape[0]
		self.height = shape[1]
		self.depth = 1 if len(shape) <= 2 else shape[3]

	def forward(self, inputs, keepacts=False):
		assert inputs.shape == self.input_shape, 'Wrong input shape'
		self._forward(inputs, keepacts)

	def backward(self, gradient, keepgrad=True):
		assert gradient.shape == self.output_shape, 'Wrong gradient shape'
		self._backward(gradient, keepgrad)

	def update_parameters():
		""" To be implemented in child class (if needed). """
		pass

	def _forward(self, inputs, keepacts):
		""" To be implemented in child class. """
		raise NotImplementedError

	def _backward(self, gradient, keepgrad):
		""" To be implemented in child class. """
		raise NotImplementedError


class ConvLayer(Layer):

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

	def update_parameters():
		pass


class PoolLayer(Layer):

	def __init__(self):
		pass

	def _forward(self, inputs, keepacts=False):
		pass

	def _backward(self, gradient, keepgrad=True):
		pass

	def update_parameters():
		pass


class ReluLayer(Layer):

	def __init__(self):
		pass

	def _forward(self, inputs, keepacts=False):
		pass

	def _backward(self, gradient, keepgrad=True):
		pass

	def update_parameters():
		pass

class FCLayer():

	def __init__(self):
		pass

	def _forward(self, inputs, keepacts=False):
		pass

	def _backward(self, gradient, keepgrad=True):
		pass

	def update_parameters():
		pass


class SoftmaxLayer(Layer):

	def __init__(self):
		pass

	def _forward(self, inputs, keepacts=False):
		pass

	def _backward(self, gradient, keepgrad=True):
		pass

	def update_parameters():
		pass

