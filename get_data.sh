#!/bin/bash

URLS=(
"http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
"http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
"http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
"http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
"http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
)

mkdir data
for URL in ${URLS[*]}; do
	wget $URL -P data/
done
for F in data/*.gz ; do
	gzip -d $F
done

tar -xf data/*.tar -C data/
rm data/*.tar
