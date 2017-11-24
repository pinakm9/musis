import tensorflow as tf
import numpy as np
from utility import * 
import process as pr
# Data sets
gtzan = pr.MusicDB(p2_train, p2_train_label, p2_test, p2_test_label)

dataset = tf.data.Dataset.from_tensor_slices((gtzan.train.music, gtzan.train.labels))

training_epochs = 25
learning_rate = 0.01

batch_size = 100

display_step = 1
#dataset = dataset.batch(batch_size)

print dataset.train.next_batch(batch_size)
"""
x = tf.placeholder("float", [None, 13*fpf])
y = tf.placeholder("float", [None, 10])
W = tf.Variable(tf.zeros([13*fpf, 10]))
b = tf.Variable(tf.zeros([10]))
evidence = tf.matmul(x, W) + b
activation = tf.nn.softmax(tf.matmul(x, W) + b)
cross_entropy = y*tf.log(activation)
cost = tf.reduce_mean\

         (-tf.reduce_sum\

           (cross_entropy, reduction_indices=1))
optimizer = tf.train.GradientDescentOptimizer\

 (learning_rate).minimize(cost)
avg_set = []

epoch_set=[]
init = tf.initialize_all_variables()
with tf.Session() as sess:
	sess.run(init)
	for epoch in range(training_epochs):
		avg_cost = 0.
		total_batch = int(gtzan.train.music.shape[0]/batch_size)
		for i in range(total_batch):
			batch_xs, batch_ys = \
			dataset.next_batch(batch_size)
"""