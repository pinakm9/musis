import tensorflow as tf
from utility import *
import process as pr
import numpy as np

def net(genres_to_keep = '0123456789', epochs = 80, learning_rate = 0.001, hnodes = [500, 100, 10], use_layers = 2):
	"""
	Neural network function, returns True if training was successful and False otherwise. In case training was 
    unsuccessful (Method strayed from minima) try decreasing learning_rate. hnodes contains number of nodes in 
    hidden layers, use_layers specifies if 2 or 3 hidden layers are to be used. genres_to_keep is a string containing 
    the numeric ids of genres we'd like experiment on. net computes a model and saves it in model/model(use_layers) 
    folder.
	"""
	# Prepare data
	gtzan = pr.MusicDB(p2_train, p2_train_label, p2_test, p2_test_label)
	genres_to_remove = gmap(genres_to_keep, rest = True)
	gtzan.remove_genres(genres_to_remove)
	batch_size = 100
	genre = len(genres) - len(genres_to_remove)
	features = gtzan.train.music[0].shape[0]
	if use_layers == 2:
		hnodes[2] = genre
		save_path = p2_m2
	else:
		save_path = p2_m3
	# Declare the training data placeholders, input x = mfcc array
	x = tf.placeholder(tf.float32, [None, features])
	# Declare the output data placeholder - number of digits = genre
	y = tf.placeholder(tf.float32, [None, genre])
	# Declare the weights connecting the input to the hidden layer 1
	W1 = tf.Variable(tf.random_normal([features, hnodes[0]], stddev=0.03), name='W1')
	b1 = tf.Variable(tf.random_normal([hnodes[0]]), name='b1')
	# Weights connecting the hidden layer 1 to the hidden layer 2
	W2 = tf.Variable(tf.random_normal([hnodes[0], hnodes[1]], stddev=0.04), name='W2')
	b2 = tf.Variable(tf.random_normal([hnodes[1]]), name='b2')
	# Weights connecting the hidden layer 2 to the hidden layer 3
	W3 = tf.Variable(tf.random_normal([hnodes[1], hnodes[2]], stddev=0.05), name='W3')
	b3 = tf.Variable(tf.random_normal([hnodes[2]]), name='b3')

	# Calculate the output of the hidden layers
	hidden_out1 = tf.add(tf.matmul(x, W1), b1)
	hidden_out1 = tf.nn.relu(hidden_out1)

	hidden_out2 = tf.add(tf.matmul(hidden_out1, W2), b2)
	hidden_out2 = tf.nn.relu(hidden_out2)
	
	if use_layers == 3:
		W4 = tf.Variable(tf.random_normal([hnodes[2], genre], stddev = 0.03), name = 'W4')
		b4 = tf.Variable(tf.random_normal([genre]), name = 'b4')
		hidden_out3 = tf.add(tf.matmul(hidden_out2, W3), b3)
		hidden_out3 = tf.nn.relu(hidden_out3)
		y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out3, W4), b4))
	else:
		y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out2, W3), b3))
	# Calculate softmax activated output layer	
	y_c = tf.clip_by_value(y_, 1e-10, 0.9999999)
	cross_entropy = -tf.reduce_mean(tf.reduce_sum(y*tf.log(y_c) + (1-y)*tf.log(1-y_c), axis=1))
	# Add an optimizer
	optimizer = tf.train.RMSPropOptimizer(learning_rate = learning_rate).minimize(cross_entropy)
	# Setup the initialization operator
	init_op = tf.global_variables_initializer()
	# Add ops to save and restore all the variables.
	saver = tf.train.Saver()
	# Define an accuracy assessment operation
	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	# Start the session
	cost0, fails = 1e9, 0
	with tf.Session() as sess:
	   # Initialize the variables
		sess.run(init_op)
		total_batch = int(len(gtzan.train.labels)/batch_size)
		for epoch in range(epochs):
			avg_cost = 0
			for i in range(total_batch):
				batch_x, batch_y = gtzan.train.next(batch_size)
				_, c = sess.run([optimizer, cross_entropy], feed_dict={x: batch_x, y: batch_y})
				avg_cost += c / total_batch
			# Fail-safe if the minimization stagnates
			if cost0 < avg_cost and avg_cost > 1e-6:
				fails += 1
				if fails == 50:
					print("Method strayed away from minima")
					return False # Training was unsuccessful
			else:
				cost0 = avg_cost
				fails = 0
			print("Epoch: {}, cost = {:.9f}".format((epoch + 1), avg_cost))
		train_acc = sess.run(accuracy, feed_dict={x: gtzan.train.music, y: gtzan.train.labels})
		# Print results
		test_acc = sess.run(accuracy, feed_dict={x: gtzan.test.music, y: gtzan.test.labels})
		print('Accuracy on training data: {:.2f}%'.format(100*train_acc))
		print('Accuracy on test data: {:.2f}%'.format(100*test_acc))
		# Store the model for future use
		saver.save(sess, save_path)
	# Store the results in a text file
	g2k = list(map(int, list(genres_to_keep)))
	g2k.sort()
	g2k = ''.join([str(i) for i in g2k])
	with open(p2_results, 'a') as file:
		file.write('{}  {}  {:.4f}\t{:.2f}\t{:.2f}\t{}  {}\n'\
			.format(g2k, epochs, learning_rate, train_acc*100, test_acc*100, hnodes, use_layers))
	return True # Training was successful 

# Usage: the example below creates a prdiction model and runs it on the 2-genre set {classical, metal} for both
# the training data and test data with 2 hidden layers with the 1st one having 200 nodes and 2nd one having 100
# nodes, as we are only using 2 hidden layers, last entry of hnode list is ignored 
net('16', 300, 0.0025, [200, 100, 100], use_layers = 2)