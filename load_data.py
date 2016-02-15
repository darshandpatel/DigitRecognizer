from numpy.core.shape_base import vstack
import numpy as np
import random

class LoadData:

    def __init__(self):
        self.train_features = None
        self.test_features = None
        self.train_labels = None
        self.nbr_of_train_dp = 0
        self.nbr_of_test_dp = 0
        self.count = 0
        self.test_count = 0

    def load_train_data(self, file_path, file_name):
        label = list()
        features = list()

        full_path = file_path + file_name
        count = 0
        with open(full_path, 'r') as file_pointer:

            for file_line in file_pointer:
                if (count != 0) and ("," in file_line):
                    values = np.zeros(10)
                    line_parts = file_line.split(",")
                    values[int(line_parts[0])] = 1
                    label.append(values)
                    features.append(np.array(map(float, line_parts[1:]))/255)
                count = count + 1

        self.train_features, self.train_labels = np.array(features, dtype='float32'), \
                                                 np.array(label, dtype='float32')
        self.nbr_of_train_dp = self.train_features.shape[0]
        return self.train_features, self.train_labels

    def get_train_batch(self, batch_size):

        #print self.count
        if batch_size > self.nbr_of_train_dp:
            return None, None
        elif (self.nbr_of_train_dp - self.count) > batch_size:
            batch_features = self.train_features[self.count:(self.count + batch_size), ]
            batch_labels = self.train_labels[self.count:(self.count + batch_size), ]
            self.count = self.count + batch_size
	        #print(batch_labels[15])
            return batch_features, batch_labels
        else:
            nbr_gap = batch_size - (self.nbr_of_train_dp - self.count)
            batch_features = np.vstack((self.train_features[self.count:self.nbr_of_train_dp, ],
                                    self.train_features[0:nbr_gap, ]))
            batch_labels = np.vstack((self.train_labels[self.count:self.nbr_of_train_dp, ],
                                  self.train_labels[0:nbr_gap, ]))
            self.count = 0
            return batch_features, batch_labels

    def get_test_batch(self, batch_size):

        if batch_size > self.nbr_of_test_dp:
            return None
        elif (self.nbr_of_test_dp - self.test_count) >= batch_size:
            batch_features = self.test_features[self.test_count:(self.test_count + batch_size), ]
            self.test_count = self.test_count + batch_size
            return batch_features
        else:
            return None

    def load_test_data(self, file_path, file_name):

        features = list()
        full_path = file_path + file_name

        with open(full_path, 'r') as file_pointer:
            count = 0
            for file_line in file_pointer:
                if (count != 0) and ("," in file_line):
                    line_parts = file_line.split(",")
                    features.append(np.array(map(float, line_parts))/255)
                count = count + 1

        self.test_features = np.array(features, dtype='float32')
        print(type(self.test_features))
        self.nbr_of_test_dp = self.test_features.shape[0]
        return self.test_features

    @staticmethod
    def random_sample(set_x, set_y, sample_size):

        size = set_x.shape[0]
        random_numbers = random.sample(xrange(0, size), sample_size)
        return set_x[random_numbers, ], set_y[random_numbers, ]


if __name__ == "__main__":
    load_data = LoadData()
    test_set_x = load_data.load_test_data("/home/darshan/Documents/DigitRecognizer/MNIST_data/",
                                          "test.csv")
    print(test_set_x.shape)
    nbr_of_test_batches = 10
    batch_size = load_data.nbr_of_test_dp / nbr_of_test_batches
    print('Batch size '+ str(batch_size))
    sum_data = 0
    for j in xrange(nbr_of_test_batches):
        test_batch = load_data.get_test_batch(batch_size)
        print type(test_batch)
        sum_data = sum_data + test_batch.shape[0]
    print sum_data



