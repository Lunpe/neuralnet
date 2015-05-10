import numpy as np
import nnutils


class FCNetwork():
	""" A fully-connected and simplistic neural network.
		Assumes ReLu activation function and softmax as loss function."""

	def __init__(self, layout, learn_rate=0.01, batch_size=50, n_epochs=10):
		self.layout = layout
		self.weights = np.array([np.random.randn(l1, l2) * np.sqrt(2.0/l1)
			for l1, l2 in zip(layout[:-1], layout[1:])])
		self.biases = np.array([np.zeros(l) for l in layout[1:]])
		self.fired = np.array([None for l in layout])
		# Hyperparameters:
		self.learn_rate = learn_rate
		self.batch_size = batch_size
		self.n_epochs = n_epochs

	def train(self, data):
		""" data has shape (n_images, pixels) """
		for epoch in xrange(self.n_epochs):
			print('epoch ' + str(epoch+1) + ' / ' + str(self.n_epochs))
			for n in xrange(0, len(data[0]), self.batch_size):
				self._train_batch((data[0][n: n + self.batch_size],
						data[1][n: n + self.batch_size]))

	def _train_batch(self, batch):
		activs = np.array([None for l in self.layout])
		# Forward pass
		activs[0] = batch[0]
		for i, (w, b) in enumerate(zip(self.weights, self.biases)[:-1]):
			# Using ReLu activation function
			activs[i+1] = np.maximum(0, np.dot(activs[i], w) + b)
		activs[-1] = np.dot(activs[-2], self.weights[-1]) \
				+ self.biases[-1] # No activation func for the output layer
		# Loss gradient computation (Softmax)
		grad = np.array([np.exp(out) / np.sum(np.exp(out))
						for out in activs[-1]])
		for i, c in enumerate(batch[1]):
			grad[i][c] -= 1
		# Backprop
		dw = [np.zeros(w.shape) for w in self.weights]
		db = [np.zeros(b.shape) for b in self.biases]
		for l in xrange(1, len(self.weights)+1): # ReLu makes it easy
			dw[-l] = np.dot(activs[-(l+1)].T, grad) / self.batch_size
			db[-l] = np.mean(grad, axis=0)
			grad = np.dot(grad, self.weights[-l].T)
		# Parameters updates
		for i in xrange(len(self.weights)):
			self.weights[i] -= self.learn_rate * dw[i]
			self.biases[i] -= self.learn_rate * db[i]

	def predict(self, datum):
		for i, (w, b) in enumerate(zip(self.weights, self.biases)[:-1]):
			# Using ReLu activation function
			datum = np.maximum(0, np.dot(datum, w) + b)
		# No activation func for the output layer
		return np.dot(datum, self.weights[-1]) + self.biases[-1]

	def test(self, data):
		""" data has shape (n_images, pixels) """
		res = [(np.argmax(self.predict(im)), c)
			for im, c in zip(data[0], data[1])]
		return sum(int(x == y) for (x, y) in res)


if __name__ == '__main__':
	tr_d, te_d = nnutils.load_final_data()
	net = FCNetwork([len(tr_d[0][0]), 100, 10])
	net.train(tr_d)
	res = net.test(te_d)
	print(str(res) + ' / ' + str(len(te_d[0])))



