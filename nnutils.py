import os
import struct
import numpy as np

from array import array as pyarray


def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def load_mnist(dataset="training", digits=np.arange(10), path="./data"):
    """
    Loads MNIST files into 2D numpy arrays

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

    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = len(ind)

    images = np.zeros((N, rows * cols), dtype=np.uint8)
    labels = np.zeros((N), dtype=np.int8)
    for i in range(len(ind)):
        images[i] = np.array(img[ind[i]*rows*cols : (ind[i]+1)*rows*cols])
        labels[i] = lbl[ind[i]]

    return images, labels

def preprocess(data):
	""" Scales the images pixels between 0 and 1."""
	return (data[0] / 255.0, data[1])

def cross_validation_data(n_validation=10000):
	""" Assuming we're using the mnist databse.
	n_validation is the number of validation data wanted from the 60000
	total training data."""
	d = preprocess(load_mnist('training'))
	i = np.random.random_integers(0, len(d[0]) - n_validation)

	# Not truly random but whatever
	training = (np.concatenate((d[0][:i], d[0][i+n_validation:])),
		np.concatenate((d[1][:i], d[1][i+n_validation:])))
	validation = (d[0][i:i+n_validation], d[1][i:i+n_validation])
	return training, validation




