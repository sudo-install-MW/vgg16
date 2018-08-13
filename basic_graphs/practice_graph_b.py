import tensorflow as tf

b = tf.constant(2)
a = tf.constant(90)

c = tf.multiply(b, a)
c = tf.cast(c, dtype=tf.float32)
d = tf.sin(c)
d = tf.cast(d, dtype=tf.int32)
e = tf.div(d, b)

sess = tf.Session()
print(sess.run(e))