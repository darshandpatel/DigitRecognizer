import tensorflow.examples.tutorials.mnist.input_data as input_data
import tensorflow as tf
from load_data import LoadData
import numpy as np


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# Load the Digit DataSet
load_data = LoadData()
train_set_x, train_set_y = load_data.load_train_data("/home/darshan/Documents/DigitRecognizer/MNIST_data/",
                                                    "train.csv")

#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
x_image = tf.reshape(x, [-1, 28, 28, 1])
y_ = tf.placeholder(tf.float32, [None, 10])

# First Layer of Convnet
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# 28 x 28 -> 24 x 24
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# 24 x 24 -> 12 x 12
h_pool1 = max_pool_2x2(h_conv1)


W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

# 12 x 12 -> 8 x 8
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
#8 x 8 -> 4 x 4
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# Loss Function : Cross Entropy
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess = tf.InteractiveSession()

sess.run(tf.initialize_all_variables())

for i in range(20000):
    batch_xs, batch_ys = load_data.get_train_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
        x: batch_xs, y_: batch_ys, keep_prob: 1.0})
    print "step %d, training accuracy %g" % (i, train_accuracy)
    train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})

#print "test accuracy %g"%accuracy.eval(feed_dict={
#    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})

test_set_x = load_data.load_test_data("/home/darshan/Documents/DigitRecognizer/MNIST_data/",
                                      "test.csv")
print(test_set_x.shape)
nbr_of_test_batches = 10
batch_size = load_data.nbr_of_test_dp / nbr_of_test_batches
for j in xrange(nbr_of_test_batches):
    test_batch = load_data.get_test_batch(batch_size)
    if test_batch is not None:
        y_predict = tf.argmax(y_conv, 1)
        result_value = sess.run(y_predict, feed_dict={x: test_batch, keep_prob: 1.0})
        result_label = xrange((batch_size * j) + 1, (batch_size * (j + 1)) + 1)
        z = np.array(zip(result_label, result_value), dtype=[('ImageId', int), ('Label', int)])
        np.savetxt('result_cnn' + str(j) + '.csv', z, fmt='%i,%i')

sess.close()
