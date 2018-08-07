import tensorflow as tf
from utils import MNIST

data_dir = '/MNIST_data'
num_steps = 1000
mini_batch = 100

mnist = MNIST()

X_train, X_label = mnist.train_set()
X_test, X_test_label = mnist.test_set()

g = tf.Graph()

with g.as_default():

    with tf.name_scope(name="Inputs") as scope:
        x = tf.placeholder(tf.float32, [None, 784])
        W = tf.Variable(tf.zeros([784, 10]))

    y_true = tf.placeholder(tf.float32, [None, 10], name="output_layer")


    with tf.name_scope(name="Training") as scope:
        y_pred = tf.matmul(x, W, name="output")
        with tf.name_scope(name="Loss_function") as scope:
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true))
        with tf.name_scope(name="optimizer") as scope:
            gd_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    with tf.name_scope(name="Inference") as scope:
        correct_mask = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        writer = tf.summary.FileWriter('./logs', sess.graph)

        for i in range(num_steps):
            #batch_xs, batch_ys = data.train.next_batch(mini_batch)
            sess.run(gd_step, feed_dict={x:X_train, y_true: X_label})
            print("training batch {}".format(i))
        accuracy = sess.run(accuracy, feed_dict={x:X_test, y_true: X_test_label})
        print("Accuracy of the model is :", accuracy)

