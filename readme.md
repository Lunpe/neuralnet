A simple convolutional neural network
-------------------------------------
(written for educational purpose)

This is a very simple package to create neural networks. It is set
to learn images from the [MNIST](http://yann.lecun.com/exdb/mnist/]) database and [CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar.html) dataset.

The only depedencies are numpy and scipy, and I hope to get rid of the latter soon.

It uses momentum coupled with softmax to train.

Using it is easy:
```python
from neuralnets import neuralnet, layers

net = neuralnets.NeuralNetwork(tr_d[0].shape)
net.add_layer(layers.ConvLayer, n_filters=32)
net.add_layer(layers.BiasLayer)
net.add_layer(layers.ReLuLayer)
net.add_layer(layers.ConvLayer, n_filters=16)
net.add_layer(layers.BiasLayer)
net.add_layer(layers.ReLuLayer)
net.add_layer(layers.FCLayer, n_neurons=10)

net.train(images, labels, n_epochs=5, momentum=0.8)
answer = net.predict(image_whatisit)
```
But as a user, you'd better go for something else like [caffe](https://github.com/BVLC/caffe).
