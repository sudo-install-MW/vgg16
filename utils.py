# script for preprocessing and fetching image data
from tensorflow.examples.tutorials.mnist import input_data

class MNIST():
    def __init__(self):
        pass
    
    def train_set(self):
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        #print("Type of the train images ",type(mnist.train.images))
        #print("Dimention of the train images", mnist.train.images.shape)
        return mnist.train.images, mnist.train.labels
    
    def test_set(self):
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        return mnist.test.images, mnist.test.labels