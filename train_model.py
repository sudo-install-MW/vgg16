# script to train model
from utils import MNIST
from network import network
#score = model.evaluate(X_test, Y_test, verbose=0)

def train():
    mnist = MNIST()
    X_train, X_label = mnist.train_set()
    Y_test, Y_label = mnist.test_set()

    train_network = network()
    train_network.fc_network(X_train, X_label, Y_test, Y_label)

train()