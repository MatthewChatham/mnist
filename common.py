from tensorflow.examples.tutorials.mnist import input_data

def get_mnist():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    xtrain = mnist.train.images
    ytrain = mnist.train.labels

    xtest = mnist.test.images
    ytest = mnist.test.labels

    xval = mnist.validation.images
    yval = mnist.validation.labels
    
    return {'x': {'train': xtrain, 'test': xtest, 'val': xval}, 'y': {'train': ytrain, 'test': ytest, 'val': yval}}