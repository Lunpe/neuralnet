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
	""" A convolutional layer. """

	def __init__(self, input_shape,
			n_filters,
			field=3,
			zero_pad=1,
			stride=1):
		""" input_shape is expected to be (N, X, X, C) """
		# TODO: some asserts to make sure consistent hyperparameters are given
		super(ConvLayer, self).__init__(input_shape)
		self.input_channels = input_shape[3] if len(input_shape) > 3 else 1
		# Hyperparameters
		self.n_filters = n_filters
		self.field = field
		self.zero_pad = zero_pad
		self.stride = stride
		 # Parameters
		self.weights = np.random.randn(n_filters, field, field, self.input_channels)
		for i in xrange(n_filters):
			self.weights[i] *= np.sqrt(2.0 / np.sum(self.weights[i].shape))
		self.output_shape = (input_shape[0], input_shape[1], input_shape[2], n_filters)

	def _forward(self, inputs, keepacts=False):
		# Hopefully taking advantage of numpy's arrays mult. optimizations
		outputs = []
		w_map = self._weight2row(self.weights)
		for x in inputs:
			x_mat = self._im2col(inp)
			outputs.append(self._col2im(np.dot(w_mat, x_mat)))
		return outputs

	def _backward(self, gradient, keepgrad=True):
		pass

	def update_parameters(self):
		pass

	def _im2col(self, x):
		""" From input_shape (X, X, C) to matrix (F*F*C, _)> """
		height_col = self.field * self.field * self.input_channels
		width_col = x.shape[0] + 2 * self.zero_pad - self.field
		width_col = width_col / self.stride + 1
		n_acts = width_col # The number of activations per line/column of x
		width_col *= width_col
		col = np.zeros((height_col, width_col))
		for ic in xrange(height_col):
			for jc in xrange(width_col):
				for c in xrange(self.input_channels):
					i = self.stride * (jc / n_acts) + ic % self.field
					j = (jc % self.field) * self.stride + ic / self.field
					i -= self.zero_pad
					j -= self.zero_pad
					if i < 0 or j < 0 or i >= x.shape[0] or j >= x.shape[1]:
						continue
					else:
						col[c*self.field*self.field + ic][jc] = x[i][j]
		return col

	def _col2out(self, col):
		""" From (K, _) to (X, X, K) """
		out = np.zeros((self.input_shape[1], self.input_shape[2], self.n_filters))
		for i in xrange(self.input_shape[1]):
			for j in xrange(self.input_shape[1]):
				for k in xrange(self.n_filters):
					out[i][j][k] = col[k][i*self.input_shape[1] + j]
		return out

	def _weight2row(self):
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
		""" input_shape is expected to be (N, X) """
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

