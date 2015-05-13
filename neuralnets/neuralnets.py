import layers
import numpy as np
import itertools as it

# TODO: Build an automated cross-validation tool for fine-tuning

class NeuralNetwork(object):
	""" Base class of a neural network.

	Uses momentum and softmax as default.
	"""

	def __init__(self, input_shape):
		""" Creates a neural network.

		input_shape: the shape of the input (num x channels x dim x dim)
		"""
		self.input_shape = input_shape
		self.layers = []

	def forward(self, x, keepacts=False):
		""" Predicts the class of x through the net., returns the predictions.

		x: the inputs (usually images) in a numpy array
		keepacts: determines if the layers will keep track of their activations
		must be set to True if backpropagation is intended
		"""
		for layer in self.layers:
			x = layer.forward(x, keepacts=keepacts)
		return x

	def backward(self, gradient):
		""" Executes the backpropagation algorithm.

		gradient: the error gradient on the output of the network
		"""
		for layer in reversed(self.layers):
			gradient = layer.backward(gradient)
		return gradient

	def predict(self, x):
		""" Predicts the class of a single datum. """
		output = self.forward([x])
		return np.argmax(output)

	def test(self, x, y):
		""" Returns the number of good predictions by the network on x and y.

		x: the inputs (images)
		y: the correponding labels
		"""
		maxs = np.argmax(self.forward(x), axis=1)
		return sum((o == y) for (o, y) in it.izip(maxs, y))

	def train(self, x,
			y,
			n_epochs=10,
			batch_size=50,
			learn_rate=0.01,
			momentum=0.5):
		""" Trains the network with given input and labels.

		x: the inputs in a numpy array (number x channels x width x height)
		y: the correponding labels
		n_epochs: the number of epochs used for training
		batch_size: the size of a mini batch used for training
		learn_rate: well, the learning rate
		momentum: momentum used for parameter updates

		May be reimplemented in a child class
		"""
		# TODO: Use momentum update for the parameters as default ?
		# (actually: having a default model for parameter updates)
		for epoch in xrange(n_epochs):
			print('epoch ' + str(epoch + 1) + ' / ' + str(n_epochs))
			for i_batch in xrange(0, len(x), batch_size):
				xb = x[i_batch:i_batch+batch_size]
				yb = y[i_batch:i_batch+batch_size]

				outputs = self.forward(xb, keepacts=True)
				grad = self.loss(outputs, yb)
				self.backward(grad)
				self.update_parameters(learn_rate, momentum)

	def add_layer(self, layer_class, **kwargs):
		""" Adds a layer of the given class at the end of the net.

		The keyword arguments are passed to the class at instantiation.
		"""
		if len(self.layers) == 0:
			self.layers.append(layer_class(self.input_shape, **kwargs))
		else:
			self.layers.append(layer_class(self.layers[-1].output_shape,\
					**kwargs))
		return self

	def update_parameters(self, learn_rate, momentum):
		""" Makes the layers update their parameters. """
		for layer in self.layers:
			layer.update_parameters(learn_rate, momentum)

	def loss(self, outputs, ys):
		""" The function used as loss function by the default training algo.

		outputs: the output of the network for given inputs
		ys: the true classes for the inputs

		Default: softmax
		"""
		return softmax(self, outputs, ys)


# TODO: Find/create a place for this function (and other loss funcs)
def softmax(self, outputs, ys):
	""" A loss function. """
	grad = np.array([np.exp(o) / np.sum(np.exp(o)) for o in outputs])
	for i, c in enumerate(ys):
		grad[i][c] -= 1
	return grad

