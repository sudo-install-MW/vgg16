import tensorflow as tf
import numpy as np

x_data = np.random.randn(5, 10)
w_data = np.random.randn(10, 2)

with tf.Graph().as_default():
    x = tf.placeholder(tf.float32, (None, 10))
    w = tf.placeholder(tf.float32, (10, None))
    b = tf.fill((5,1), -1.)
    xw = tf.matmul(x, w)

    xwb = xw + b
    s = tf.reduce_max(xwb)
    with tf.Session() as sess:
        outs = sess.run(s, feed_dict={x: x_data, w: w_data})
    
    print("outs = {}".format(outs))
