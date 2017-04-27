
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf

# https://www.tensorflow.org/get_started/get_started

# A complete discussion of machine learning is out of the scope of 
# this tutorial. However, TensorFlow provides optimizers that slowly 
# change each variable in order to minimize the loss function. 
# The simplest optimizer is gradient descent. It modifies each 
# variable according to the magnitude of the derivative of loss with 
# respect to that variable. In general, computing symbolic derivatives 
# manually is tedious and error-prone. Consequently, TensorFlow 
# can automatically produce derivatives given only a description 
# of the model using the function tf.gradients. For simplicity, 
# optimizers typically do this for you. For example,


# Model parameters
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)

# Model input and output
x = tf.placeholder(tf.float32)
linear_model = W * x + b
y = tf.placeholder(tf.float32)

# loss
loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares

# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# training data
x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]

# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
for i in range(1000):
  sess.run(train, {x:x_train, y:y_train})

# evaluate training accuracy
curr_W, curr_b, curr_loss  = sess.run([W, b, loss], {x:x_train, y:y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
