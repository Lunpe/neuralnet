import layers
import numpy as np
import itertools as it

# TODO: Build an automated cross-validation tool for fine-tuning

class NeuralNetwork(object):
	""" Abstract class of a neural network.
		To use a neuralnet one must create a child class (of NeuralNetwork
		or one of its subclasses) that instantiates its own layers
		and, if wanted, learning methods. """

	def __init__(self,
			input_shape,
			n_epochs,
			batch_size,
			learn_rate,
			regu_strength):
		self.input_shape = input_shape
		self.layers = []
		self.n_epochs = n_epochs
		self.batch_size = batch_size
		self.learn_rate = learn_rate
		self.regu_strength = regu_strength

	def forward(self, x, keepacts=False):
		for layer in self.layers:
			x = layer.forward(x, keepacts=keepacts)
		return x

	def backward(self, gradient, keepgrad=True):
		for layer in reversed(self.layers):
			gradient = layer.backward(gradient, keepgrad=keepgrad)
		return gradient

	def test(self, x, y):
		maxs = np.argmax(self.forward(x), axis=1)
		return sum((o == y) for (o, y) in it.izip(maxs, y))

	def train(self, x, y):
		""" May be reimplemented in a child class. """
		# TODO: Use momentum update for the parameters as default ?
		# (actually: having a default model for parameter updates)
		for epoch in xrange(self.n_epochs):
			print('epoch ' + str(epoch + 1) + ' / ' + str(self.n_epochs))
			for i_batch in xrange(0, len(x), self.batch_size):
				xb = x[i_batch:i_batch+self.batch_size]
				yb = y[i_batch:i_batch+self.batch_size]

				outputs = self.forward(xb, keepacts=True)
				grad = self.loss(outputs, yb)
				self.backward(grad)
				self.update_parameters()

	def _add_layer(self, layer_class, **kwargs):
		if len(self.layers) == 0:
			self.layers.append(layer_class(self.input_shape, **kwargs))
		else:
			self.layers.append(layer_class(self.layers[-1].output_shape, **kwargs))
		return self

	def update_parameters(self):
		""" To be implemented in child class. """
		raise NotImplementedError

	def loss(self, outputs, ys):
		""" To be implemented in child class. """
		raise NotImplementedError


class ConvNet(NeuralNetwork):

	def __init__(self, input_shape,
			n_epochs,
			batch_size,
			learn_rate,
			regu_strength):
		super(ConvNet, self).__init__(input_shape, n_epochs,\
				batch_size, learn_rate, regu_strength)
		self.layers = []
		self._add_layer(layers.ConvLayer, n_filters=16)
		self._add_layer(layers.ReLuLayer)
		self._add_layer(layers.FCLayer, n_neurons=10)

	def update_parameters(self):
		for layer in self.layers:
			layer.update_parameters(self.learn_rate, self.regu_strength)

	def loss(self, outputs, ys):
		# Softmax function
		grad = np.array([np.exp(o)/np.sum(np.exp(o)) for o in outputs])
		for i, c in enumerate(ys):
			grad[i][c] -= 1
		return grad

class SoftmaxFCNetwork(NeuralNetwork):

	def __init__(self, input_shape, layout, n_epochs, batch_size, learn_rate, regu_strength):
		super(SoftmaxFCNetwork, self).__init__(input_shape, n_epochs, batch_size, learn_rate, regu_strength)
		for n_neurons in layout[:-1]:
			self._add_layer(layers.FCLayer, n_neurons=n_neurons)
			self._add_layer(layers.BiasLayer)
			self._add_layer(layers.ReLuLayer)
		# There's no activation function on the last layer
		self._add_layer(layers.FCLayer, n_neurons=layout[-1])
		self._add_layer(layers.BiasLayer)

	def update_parameters(self):
		for layer in self.layers:
			layer.update_parameters(self.learn_rate, self.regu_strength)

	def loss(self, outputs, ys):
		# Softmax function
		grad = np.array([np.exp(o) / np.sum(np.exp(o)) for o in outputs])
		for i, c in enumerate(ys):
			grad[i][c] -= 1
		return grad

