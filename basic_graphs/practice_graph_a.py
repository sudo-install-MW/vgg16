import tensorflow as tf

b = tf.constant(4)
a = tf.constant(5)

d = tf.add(a, b)
c = tf.multiply(a, b)

f = tf.add(c, d)
e = tf.subtract(c, d)

g = tf.divide(f, e)

sess = tf.Session()

print(sess.run(g))