import tensorflow as tf

# create graph
g = tf.Graph()

# set g as default graph and construct the graph
with g.as_default():
    x = tf.placeholder(dtype=tf.float32, shape=None, name='x')
    w = tf.Variable(2.0, name='weight')
    b = tf.Variable(1.0, name='bias')
    z = w*x + b
    init = tf.global_variables_initializer()

# execute the graph session for graph g
with tf.Session(graph=g) as sess:
    sess.run(init)
    for t in [1, 2, 3, 4, 5]:
        print(sess.run(z, feed_dict={x: t}))
