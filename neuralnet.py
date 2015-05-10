import nnutils
import layers
import numpy as np
import itertools as it

class NeuralNetwork():
	""" Abstract class of a neural network.
		To use a neuralnet one must create a child class (of NeuralNetwork
		or one of its subclasses) that instantiates its own layers. """

	def __init__(self, n_epochs, batch_size, learn_rate):
		self.layers = []
		self.n_epochs = n_epochs
		self.batch_size = batch_size
		self.learn_rate = learn_rate

	def forward(self, x, keepacts=False):
		activs = data[0]
		for layer in self.layers:
			activs = layer.forward(activs, keepacts=keepacts)
		return activs

	def backward(self, gradient, keepgrad=True):
		for layer in reversed(self.layers):
			gradient = layer.backward(gradient, keepgrad=keepgrad)
		return gradient

	def test(self, x, y):
		maxs = np.argmax(self.forward(x), axis=0)
		return sum((o == y) for (o, y) in it.izip(maxs, y))

	def train(self, x, y):
		""" May be reimplemented in a child class. """
		for epoch in xrange(self.n_epochs):
			for i_batch in xrange(0, len(x), self.batch_size):
				xb = x[i_batch:i_batch+self.batch_size]
				yb = y[i_batch:i_batch+self.batch_size]

				outputs = self.forward(xb, keepacts=True)

				# Computing the loss gradient
				losses = self.loss(outputs, yb)
				y_mat = np.zeros(losses.shape)
				for i, i_y in enumerate(yb):
					y_mat[yb[i]] = 1
				grad = losses - y_mat

				self.backward(grad)

				update_parameters()

	def update_parameters(self):
		""" To be implemented in child class. """
		raise NotImplementedError

	def loss(self, outputs, ys):
		""" To be implemented in child class. """
		raise NotImplementedError


class ConvNet(NeuralNetwork):

	def __init__(self, n_epochs, batch_size, learn_rate):
		super(ConvNet, self).__init__(n_epochs, batch_size, learn_rate)
		self.layers = []
		layers.append(layers.ConvLayer())
		layers.append(layers.ReluLayer())
		layers.append(layers.ConvLayer())
		layers.append(layers.ReluLayer())
		layers.append(layers.PoolLayer())
		layers.append(layers.FCLayer())
		layers.append(layers.SoftmaxLayer())

	def update_parameters():
		pass


