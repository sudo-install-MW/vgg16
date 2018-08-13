import tensorflow as tf

init_val = tf.random_normal((1, 5), 0, 1)
var = tf.Variable(init_val, name='var')
print("pre run: \n{}".format(var))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print("init value is ",sess.run(init))
    post_var = sess.run(var)

print("post run: \n{}".format(post_var))
