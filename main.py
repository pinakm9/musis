import tensorflow as tf
from utility import *
import process as pr

gtzan = pr.MusicDB(p2_train, p2_train_label, p2_test, p2_test_label)

# Python optimisation variables
learning_rate = 0.01
epochs = 50
batch_size = 100

# declare the training data placeholders
# input x - mfcc array
x = tf.placeholder(tf.float32, [None, 13*fpf])
# now declare the output data placeholder - 10 digits
y = tf.placeholder(tf.float32, [None, 10])
# now declare the weights connecting the input to the hidden layer
W1 = tf.Variable(tf.random_normal([13*fpf, 1000], stddev=0.03), name='W1')
b1 = tf.Variable(tf.random_normal([1000]), name='b1')
# and the weights connecting the hidden layer to the output layer
W2 = tf.Variable(tf.random_normal([1000, 100], stddev=0.03), name='W2')
b2 = tf.Variable(tf.random_normal([100]), name='b2')

W3 = tf.Variable(tf.random_normal([100, 10], stddev=0.03), name='W3')
b3 = tf.Variable(tf.random_normal([10]), name='b3')

"""W4 = tf.Variable(tf.random_normal([500, 100], stddev=0.03), name='W4')
b4 = tf.Variable(tf.random_normal([100]), name='b4')

W5 = tf.Variable(tf.random_normal([100, 10], stddev=0.03), name='W4')
b5 = tf.Variable(tf.random_normal([10]), name='b4')"""

# calculate the output of the hidden layer
hidden_out1 = tf.add(tf.matmul(x, W1), b1)
hidden_out1 = tf.nn.sigmoid(hidden_out1)

hidden_out2 = tf.add(tf.matmul(hidden_out1, W2), b2)
hidden_out2 = tf.nn.sigmoid(hidden_out2)

"""hidden_out3 = tf.add(tf.matmul(hidden_out2, W3), b3)
hidden_out3 = tf.nn.relu(hidden_out3)

hidden_out4 = tf.add(tf.matmul(hidden_out3, W4), b4)
hidden_out4 = tf.nn.relu(hidden_out4)"""
# now calculate the hidden layer output - in this case, let's use a softmax activated
# output layer
y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out2, W3), b3))
y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped)
                         + (1 - y) * tf.log(1 - y_clipped), axis=1))
# add an optimiser
optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
# finally setup the initialisation operator
init_op = tf.global_variables_initializer()

# define an accuracy assessment operation
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# start the session
with tf.Session() as sess:
   # initialise the variables
	sess.run(init_op)
	total_batch = int(len(gtzan.train.labels)/batch_size)
	for epoch in range(epochs):
		avg_cost = 0
		for i in range(total_batch):
			batch_x, batch_y = gtzan.train.next(batch_size)
			_, c = sess.run([optimiser, cross_entropy],\
	                     feed_dict={x: batch_x, y: batch_y})
			avg_cost += c / total_batch
		print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))
	print(sess.run(accuracy, feed_dict={x: gtzan.train.music, y: gtzan.train.labels}))
	print(sess.run(accuracy, feed_dict={x: gtzan.test.music, y: gtzan.test.labels}))
	print(sess.run(accuracy, feed_dict={x: [gtzan.train.music[0]], y: [gtzan.train.labels[0]]}))