import numpy as np
import tensorflow as tf

epoch=100000

x_data = np.random.randn(2000, 3)
w_real = [0.3, 0.5, 0.1]
b_real = -0.2

noise = np.random.randn(1, 2000) * 0.1
y_data = np.matmul(w_real, x_data.T) + b_real + noise

# step 1 Prep inputs
x_train = tf.placeholder(tf.float32, shape=[None, 3])
w_train = tf.Variable(dtype=tf.float32, initial_value=[[3,1,2]])
b_train = tf.Variable(0, dtype=tf.float32)
# step 2 Model
# Linear regression 
y_train = tf.matmul(x_train, tf.transpose(w_train)) + b_train
y_train = tf.transpose(y_train)
# step 3 Loss function
loss = tf.losses.mean_squared_error(y_data, y_train)

# step 4 Optimization
optimizer = tf.train.GradientDescentOptimizer(learning_rate = .0001)
train = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(epoch):
        y_out = sess.run(train, feed_dict={x_train:x_data})
        if epoch % 1 == 0:
            print(sess.run([w_train, b_train]))