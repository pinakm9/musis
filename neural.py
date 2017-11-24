import tensorflow as tf
import process as pr
from utility import *

# Import GTZAN data (Numpy format)
gtzan = pr.MusicDB(p2_train, p2_train_label, p2_test, p2_test_label)

# Parameters
learning_rate = 0.001
num_steps = 400
batch_size = 128
display_step = 20

# Network Parameters
n_input = 13*fpf # gtzan data input (img shape: 28*28)
n_classes = 10 # gtzan total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

sess = tf.Session()
# Create a dataset tensor from the music and the labels
dataset = tf.data.Dataset.from_tensor_slices((gtzan.train.music, gtzan.train.labels))
# Create batches of data
dataset = dataset.batch(batch_size)
# Create an iterator, to go over the dataset
iterator = dataset.make_initializable_iterator()
# It is better to use 2 placeholders, to avoid to load all data into memory,
# and avoid the 2Gb restriction length of a tensor.
_data = tf.placeholder(tf.float32, [None, n_input])
_labels = tf.placeholder(tf.float32, [None, n_classes])
# Initialize the iterator
sess.run(iterator.initializer, feed_dict={_data: gtzan.train.music,
                                          _labels: gtzan.train.labels})

# Neural Net Input
X, Y = iterator.get_next()

# -----------------------------------------------
# THIS IS A CLASSIC CNN 
# -----------------------------------------------
# Create model
def conv_net(x, n_classes, dropout, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        # gtzan data input is a 1-D vector of 784 features (28*28 pixels)
        # Reshape to match picture format [Height x Width x Channel]
        # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
        x = tf.reshape(x, shape=[-1, fpf, 13, 1])

        # Convolution Layer with 32 filters and a kernel size of 5
        conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

        # Convolution Layer with 32 filters and a kernel size of 5
        conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(conv2)

        # Fully connected layer (in contrib folder for now)
        fc1 = tf.layers.dense(fc1, 1024)
        # Apply Dropout (if is_training is False, dropout is not applied)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        # Output layer, class prediction
        out = tf.layers.dense(fc1, n_classes)
        # Because 'softmax_cross_entropy_with_logits' already apply softmax,
        # we only apply softmax to testing network
        out = tf.nn.softmax(out) if not is_training else out

    return out


# Because Dropout have different behavior at training and prediction time, we
# need to create 2 distinct computation graphs that share the same weights.

# Create a graph for training
logits_train = conv_net(X, n_classes, dropout, reuse=True, is_training=True)
# Create another graph for testing that reuse the same weights, but has
# different behavior for 'dropout' (not applied).
logits_test = conv_net(X, n_classes, dropout, reuse=True, is_training=False)

# Define loss and optimizer (with train logits, for dropout to take effect)
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits_train, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(logits_test, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Run the initializer
sess.run(init)

# Training cycle
for step in range(1, num_steps + 1):

    try:
        # Run optimization
        sess.run(train_op)
    except tf.errors.OutOfRangeError:
        # Reload the iterator when it reaches the end of the dataset
        sess.run(iterator.initializer,
                 feed_dict={_data: gtzan.train.music,
                            _labels: gtzan.train.labels})
        sess.run(train_op)

    if step % display_step == 0 or step == 1:
        # Calculate batch loss and accuracy
        # (note that this consume a new batch of data)
        loss, acc = sess.run([loss_op, accuracy])
        print("Step " + str(step) + ", Minibatch Loss= " + \
              "{:.4f}".format(loss) + ", Training Accuracy= " + \
              "{:.3f}".format(acc))

print("Optimization Finished!")