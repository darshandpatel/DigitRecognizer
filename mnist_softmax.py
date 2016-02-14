from twisted.protocols.loopback import _LoopbackAddress

import tensorflow.examples.tutorials.mnist.input_data as input_data
import tensorflow as tf
from load_data import LoadData
import numpy as np

# Load the Digit DataSet
load_data = LoadData()
train_set_x, train_set_y = load_data.load_train_data("/home/darshan/Documents/DigitRecognizer/MNIST_data/",
                                                     "train.csv")

#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Softmax function
y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])

# Loss Function : Cross Entropy
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

# Gradient Descent Step
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for i in range(1000):
    #batch_xs, batch_ys = mnist.train.next_batch(100)
    batch_xs, batch_ys = load_data.get_train_batch(200)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#print (sess.run(accuracy, feed_dict={x: mnist.train.images, y_: mnist.train.labels}))
#print (sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys}))


test_set_x = load_data.load_test_data("/home/darshan/Documents/DigitRecognizer/MNIST_data/",
                                                 "test.csv")
#ImageId,Label

y_predict = tf.argmax(y, 1)
result_value = sess.run(y_predict, feed_dict={x: load_data.test_features})
result_label = xrange(1, load_data.nbr_of_test_dp+1)
z = np.array(zip(result_label, result_value), dtype=[('ImageId', int), ('Label', int)])
np.savetxt('result.csv', z, fmt='%i,%i')

'''
with open('result.txt','w') as result_writer:
    for i in xrange(load_data.nbr_of_test_dp):
        y_predict = (tf.matmul(load_data.test_features[[i], :], W) + b)
        #result = sess.run(y_predict, feed_dict={x: load_data.test_features[[i], :]})
        result = sess.run(tf.argmax(y_predict, 1))[0]
        print(str(i))
        result_writer.write(str(result))
        result_writer.write("\n")

#print (sess.run(accuracy, feed_dict={x: test_set_x, y_: test_set_y}))
'''

sess.close()
