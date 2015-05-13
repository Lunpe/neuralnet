import os
import struct
import cPickle
import numpy as np

from array import array as pyarray

# ----------------------------------------------------------------------------
# MNIST methods
# ----------------------------------------------------------------------------


def load_mnist(dataset="training", digits=np.arange(10), path="./data"):
    """ Loads MNIST files into 2D numpy arrays

    Adapted from: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
	SEE: http://g.sweyla.com/blog/2012/mnist-numpy/
    """

    if dataset == "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [k for k in range(size) if lbl[k] in digits]
    N = len(ind)

    images = np.zeros((N, rows * cols), dtype=np.uint8)
    labels = np.zeros((N), dtype=np.int8)
    for i in range(len(ind)):
        images[i] = np.array(img[ind[i]*rows*cols : (ind[i]+1)*rows*cols])
        labels[i] = lbl[ind[i]]

    return images, labels


def preprocess_mnist(data):
	""" Scales the images pixels between 0 and 1."""
	return (data[0] / 255.0, data[1])


def load_cross_validation_data_mnist(n_validation=10000):
	""" Loads the mnist data for cross validation.

	n_validation is the number of validation data wanted from the 60000
	total training data.
	"""
	d = preprocess_mnist(load_mnist('training'))
	i = np.random.random_integers(0, len(d[0]) - n_validation)

	# Not truly random but whatever
	training = (np.concatenate((d[0][:i], d[0][i+n_validation:])),
		np.concatenate((d[1][:i], d[1][i+n_validation:])))
	validation = (d[0][i:i+n_validation], d[1][i:i+n_validation])
	return training, validation


def load_data_mnist():
	""" Loads the images and labels from the mnist database.

	Returns ((training_ims, training_labels), (testing_ims, testing_labels))
	"""
	tr = preprocess_mnist(load_mnist('training'))
	te = preprocess_mnist(load_mnist('testing'))
	return tr, te


# ----------------------------------------------------------------------------
# CIFAR 10 methods
# ----------------------------------------------------------------------------


def preprocess_cifar(images):
	""" Centers the data to have a mean of zero and scales it to [-1, 1] """
	processed = images.astype(np.float32)
	for i in xrange(len(processed)):
		for c in xrange(len(processed[0])):
			processed[i][c] = processed[i][c] - np.mean(processed[i][c])
			vmin = np.abs(np.min(processed[i][c]))
			vmax = np.abs(np.max(processed[i][c]))
			processed[i][c] = processed[i][c] / float(np.max([vmin, vmax]))
	return processed


def load_cross_validation_data_cifar():
	""" Loads the cifar-10 data for cross validation."""
	tr_images = np.empty((50000, 3, 32, 32))
	tr_labels = np.empty((50000))
	te_images = te_labels = []

	i_train = 0
	i_test = np.random.choice(6) + 1 # Choosing a random batch for cross valid.
	for i in xrange(1, 6):
		fname = "data/cifar-10-batches-py/data_batch_" + str(i)
		fo = open(fname, 'rb')
		raw = cPickle.load(fo)
		fo.close()
		data_3D = preprocess_cifar(raw['data'].reshape((10000, 3, 32, 32)))
		labels = np.array([np.argmax(l) for l in raw['labels']])

		if i == i_test:
			te_images = data_3D
			te_labels = raw['labels']
		else:
			tr_images[i_train:i_train+10000] = data_3D
			tr_labels[i_train:i_train+10000] = raw['labels']
			i_train += 10000

	return (tr_images, tr_labels), (te_images, te_labels)

