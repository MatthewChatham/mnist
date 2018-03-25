import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

xtrain = mnist.train.images
ytrain = mnist.train.labels

xtest = mnist.test.images
ytest = mnist.test.labels

xval = mnist.validation.images
yval = mnist.validation.labels


def show_image(idx=0, dataset='train'):

    plt.imshow(xtrain[idx].reshape(28, 28))
    plt.show()


show_image(1)
