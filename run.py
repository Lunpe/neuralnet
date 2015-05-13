import sys

from neuralnets import neuralnets, layers, nnutils

def fully_connected():
	# Fully connected network learning the mnist database
	print("Test: a fully connected network on the mnist database.")
	print("Loading the mnist database...")
	tr_d, te_d = nnutils.load_data_mnist()
	net = neuralnets.SoftmaxFCNetwork(input_shape=tr_d[0].shape,
			layout=[100, 10],
			n_epochs=5,
			batch_size=50,
			learn_rate=0.01,
			regu_strength=0)
	net.train(tr_d[0], tr_d[1])
	print("Testing...")
	res = net.test(te_d[0], te_d[1])
	print('We got ' + str(res) + '/' + str(len(te_d[0]))  + ' good answers !')

def convolutional():
	# Convolutional network learning the cifar10 database
	print("Test: a convolutional network on the cifar-10 database.")
	print("Loading the cifar-10 database...")
	tr_d, te_d = nnutils.load_cross_validation_data_cifar()
	net = neuralnets.ConvNet(input_shape=tr_d[0].shape,
			n_epochs=1,
			batch_size=20,
			learn_rate=0.01,
			regu_strength=0.001)
	net.train(tr_d[0], tr_d[1])
	print("Testing...")
	res = net.test(te_d[0], te_d[1])
	print('We got ' + str(res) + '/' + str(len(te_d[0]))  + ' good answers !')

if __name__ == "__main__":
	if len(sys.argv) > 1:
		if sys.argv[1] == "help":
			print("Options:")
			print("conv: trains a convolutional network on the\
					cifar-10 database")
			print("fc: trains a fully connected network on the MNIST database")
			print("The default option is fc.")
		if sys.argv[1] == "conv":
			convolutional()
		if sys.argv[1] == "fc":
			fully_connected()
	else:
		fully_connected()

