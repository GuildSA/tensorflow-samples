
# https://github.com/tensorflow/tensorflow/issues/7778
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf

# https://www.tensorflow.org/get_started/get_started

# A computational graph is a series of TensorFlow operations arranged 
# into a graph of nodes. Let's build a simple computational graph. 
# Each node takes zero or more tensors as inputs and produces a tensor 
# as an output. One type of node is a constant. Like all TensorFlow 
# constants, it takes no inputs, and it outputs a value it stores 
# internally. We can create two floating point Tensors node1 and node2 
# as follows:

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0) # This will also create a tf.float32 implicitly.

print(node1)
print(node2)

# Notice that printing the nodes does not output the values 3.0 and 4.0
# as you might expect. Instead, they are nodes that, when evaluated, 
# would produce 3.0 and 4.0, respectively. To actually evaluate the nodes, 
# we must run the computational graph within a session. A session 
# encapsulates the control and state of the TensorFlow runtime.

sess = tf.Session()
print( sess.run([node1, node2]) )


# We can build more complicated computations by combining Tensor nodes 
# with operations (Operations are also nodes.). For example, we can 
# add our two constant nodes and produce a new graph as follows:

node3 = tf.add(node1, node2)
print("node3: ", node3)
print("sess.run(node3): ", sess.run(node3))


# As it stands, this graph is not especially interesting because it 
# always produces a constant result. A graph can be paramaterized to 
# accept external inputs, known as placeholders. A placeholder is a 
# promise to provide a value later.

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

adder_node = a + b  # + provides a shortcut for tf.add(a, b)


# The preceding three lines are a bit like a function or a lambda in 
# which we define two input parameters (a and b) and then an operation 
# on them. We can evaluate this graph with multiple inputs by using 
# the feed_dict parameter to specify Tensors that provide concrete 
# values to these placeholders:

print( sess.run(adder_node, { a:3,     b:4.5   }) )
print( sess.run(adder_node, { a:[1,3], b:[2,4] }) )

# We can make the computational graph more complex by adding another
# operation. For example,

add_and_triple = adder_node * 3.

print( sess.run(add_and_triple, {a: 3, b:4.5}) )

# In machine learning we will typically want a model that can take 
# arbitrary inputs, such as the one above. To make the model trainable, 
# we need to be able to modify the graph to get new outputs with the 
# same input. Variables allow us to add trainable parameters to a graph. 
# They are constructed with a type and initial value:

W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)

linear_model = W * x + b

# Constants are initialized when you call tf.constant, and their value 
# can never change. By contrast, variables are not initialized when you 
# call tf.Variable. To initialize all the variables in a TensorFlow 
# program, you must explicitly call a special operation as follows:

init = tf.global_variables_initializer()
sess.run(init)

# It is important to realize init is a handle to the TensorFlow sub-graph 
# that initializes all the global variables. Until we call sess.run, the 
# variables are uninitialized.

# Since x is a placeholder, we can evaluate linear_model for several values
# of x simultaneously as follows:

print( sess.run( linear_model, {x:[1,2,3,4]} ) )

# We've created a model, but we don't know how good it is yet. To evaluate 
# the model on training data, we need a y placeholder to provide the desired 
# values, and we need to write a loss function.

# A loss function measures how far apart the current model is from the 
# provided data. We'll use a standard loss model for linear regression, 
# which sums the squares of the deltas between the current model and the 
# provided data. linear_model - y creates a vector where each element is 
# the corresponding example's error delta. We call tf.square to square 
# that error. Then, we sum all the squared errors to create a single scalar 
# that abstracts the error of all examples using tf.reduce_sum:

# https://www.youtube.com/watch?v=zPG4NjIkCjc&list=PLF596A4043DBEAE9C
# https://www.youtube.com/watch?v=JvS2triCgOY
# https://www.youtube.com/watch?v=w2FKXOa0HGA

y = tf.placeholder(tf.float32)

squared_deltas = tf.square(linear_model - y)

loss = tf.reduce_sum(squared_deltas)

print( sess.run( loss,                # loss function
	             {x:[1,2,3,4],        # input to model
	              y:[0,-1,-2,-3]}) )  # desired values

# producing the loss value of: 23.66

# We could improve this manually by reassigning the values of W and b 
# to the perfect values of -1 and 1. A variable is initialized to the 
# value provided to tf.Variable but can be changed using operations 
# like tf.assign. For example, W=-1 and b=1 are the optimal parameters 
# for our model. We can change W and b accordingly:

fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])

sess.run( [fixW, fixb] )

print(sess.run( loss,               # loss function
	            {x:[1,2,3,4],       # input to model
	             y:[0,-1,-2,-3]}))  # desired values
 
sess.close()

# The final print shows the loss now: 0.0

# We guessed the "perfect" values of W and b, but the whole point of 
# machine learning is to find the correct model parameters automatically. 
# We will show how to accomplish this in the next section.









