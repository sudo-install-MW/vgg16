import tensorflow as tf

a = tf.constant(5)
b = tf.constant(2)
c = tf.constant(3)

e = tf.add(c, b)
d = tf.multiply(a, b)

f = tf.subtract(d, e)

sess = tf.Session()
print(sess.run(f))
sess.close()
