# """A very simple MNIST classifier.
# See extensive documentation at
# http://tensorflow.org/tutorials/mnist/beginners/index.md
# """

import tensorflow as tf

# https://www.tensorflow.org/get_started/mnist/beginners

###########################
# Data Setup...

# The MNIST data is hosted on Yann LeCun's website.
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets( "MNIST_data/", one_hot=True )


###########################
# Model Setup...

# The variable x is a placeholder, a value that we'll input when we ask TensorFlow
# to run a computation. We want to be able to input any number of MNIST images, each
# flattened into a 784-dimensional vector. We represent this as a 2-D tensor of
# floating-point numbers, with a shape [None, 784]. (Here None means that a
# dimension can be of any length.)

x = tf.placeholder( tf.float32, [None, 784] )

# Create the weights and biases for our model.

W = tf.Variable( tf.zeros( [784, 10] ) )
b = tf.Variable( tf.zeros( [10] ) )

# We can now implement our model. First, we multiply x by W with the expression
# tf.matmul(x, W). This is flipped from when we multiplied them in our
# equation, where we had Wx, as a small trick to deal with x being a 2D tensor
# with multiple inputs. We then add b, and finally apply tf.nn.softmax.

y = tf.nn.softmax( tf.matmul( x, W ) + b )


###########################
# Training Setup...

# In order to train our model, we need to define what it means for the model to be good.
# Well, actually, in machine learning we typically define what it means for a model to
# be bad. We call this the cost, or the loss, and it represents how far off our model
# is from our desired outcome. We try to minimize that error, and the smaller the error
# margin, the better our model is.

# One very common, very nice function to determine the loss of a model is called
# "cross-entropy." To implement cross-entropy we need to first add a new placeholder
# to input the correct answers:

y_ = tf.placeholder( tf.float32, [None, 10] )

# Then we can implement the cross-entropy function:
# First, tf.log computes the logarithm of each element of y. Next, we multiply each element
# of y_ with the corresponding element of tf.log(y). Then tf.reduce_sum adds the elements
# in the second dimension of y, due to the reduction_indices=[1] parameter.
# Finally, tf.reduce_mean computes the mean over all the examples in the batch.

cross_entropy = tf.reduce_mean( -tf.reduce_sum( y_ * tf.log(y), reduction_indices=[1] ) )

# Ask TensorFlow to minimize cross_entropy using the gradient descent algorithm with a
# learning rate of 0.5. This will use a back propagation algorithm to efficiently determine
# how your variables affect the loss you ask it to minimize.

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

###########################
# Training Session...

# We can now launch the model in an InteractiveSession:

sess = tf.InteractiveSession()

# First, we have to create an operation to initialize the variables we created earlier:

tf.global_variables_initializer().run()

# Now, we do training! We'll run the training step 1000 times! Each step of the loop,
# we get a "batch" of 100 random data points from our training set.
# We run train_step feeding in the batches data to replace the placeholders.
for _ in range( 1000 ):
    batch_xs, batch_ys = mnist.train.next_batch( 100 )
    sess.run( train_step, feed_dict={x: batch_xs, y_: batch_ys} )


###########################
# Model Evaluation...

# Well, first let's figure out where we predicted the correct label. tf.argmax is an
# extremely useful function which gives you the index of the highest entry in a tensor
# along some axis. For example, tf.argmax(y,1) is the label our model thinks is most
# likely for each input, while tf.argmax(y_,1) is the correct label. We can use tf.equal
# to check if our prediction matches the truth.

correct_prediction = tf.equal( tf.argmax( y, 1 ), tf.argmax( y_, 1 ) )

# THis gives us a list of booleans. To determine what fraction are correct, we cast to
# floating point numbers and then take the mean. For example, [True, False, True, True]
# would become [1,0,1,1] which would become 0.75.

accuracy = tf.reduce_mean( tf.cast( correct_prediction, tf.float32 ) )

# Finally, we ask for our accuracy on our test data.

print( sess.run( accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels} ) )

