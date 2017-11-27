import tensorflow as tf
import numpy as np
from utility import * 
import process as pr

def identify(files, genres_to_keep, hnodes, use_layers):
	"""
	Identifies genre using a saved model, files is a list of paths to tracks we'd like to classify
	hnodes must be same as the last computed model, to know correct hnodes take a look at the results.txt file
	e.g if use_layers is 3 then the correct hnodes can be found in the last line in results.txt corresponding
	to use_layers = 3 (last column of results.txt)
	"""
	music, labels = [], []
	for file in files:	
		m, l = pr.quantum(file, genres_to_keep)
		music.append(m)
		labels.append(l)
	genre = len(genres_to_keep)
	features = music[0].shape[0]
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
	# Setup the initialization operator
	init_op = tf.global_variables_initializer()
	# Add ops to save and restore all the variables.
	saver = tf.train.Saver()
	# Define an accuracy assessment operation
	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	with tf.Session() as sess:
		# Restore variables from disk
		saver.restore(sess, save_path)
		test_acc = sess.run(accuracy, feed_dict={x: music, y: labels})
		print('Accuracy on test data: {:.2f}%'.format(100*test_acc))

identify(["./../data/genres_wav/blues/blues.00092.wav",\
"./../data/genres_wav/blues/blues.00092.wav"], '02', [500, 100, 10], 2)

