import numpy as np
import nnutils

class FCNetwork():
	""" A fully-connected neural network.
		Assumes NeLu activation function. """

	def __init__(self, layout, learn_rate=0.01, batch_size=20, n_epochs=10):
		self.weights = None
		self.biases = None
		self.fired = None
		# Hyperparameters:
		self.learn_rate = 0.01
		self.batch_size = 20
		self.n_epochs = 10

	def train(self, data):
		for epoch in xrange(self.n_epochs):
			for n in xrange(0, len(data) - self.batch_size, self.batch_size):
				train_batch(data[n, n + self.batch_size])

	def train_batch(self, batch):
		# Forward pass
		# Loss gradient computation
		# Backward pass
		# Parameters updates
		pass

	def test(self, data):
		pass


if __name__ == '__main__':
	tr_d, te_d = nnutils.cross_validation_data()
	net = FCNetwork([len(tr_d[0][0]), 20, 10])
	net.train(tr_d)



