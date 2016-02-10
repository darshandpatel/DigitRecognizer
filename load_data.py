from theano import *
import theano.tensor as T
import numpy as np


class LoadData:

    def __init__(self, file_path, file_name):
        self.file_path = file_path
        self.file_name = file_name
        self.features = list()
        self.label = list()

    def load_data(self):

        full_path = self.file_path + self.file_name

        with open(full_path, 'r') as file_pointer:

            count = 0
            for file_line in file_pointer:
                if (count != 0) and ("," in file_line):
                    line_parts = file_line.split(",")
                    self.label.append(float(line_parts[0]))
                    self.features.append(np.reshape(np.array(line_parts[1:]), (28, 28)))
                count =+ 1
        train_set_x = theano.shared(numpy.asarray(self.features, dtype=theano.config.floatX))
        train_set_y = T.cast(theano.shared(np.asarray(self.label,
                                                      dtype=theano.config.floatX)), 'int32')
        return train_set_x, train_set_y

if __name__ == "__main__":
    load_data = LoadData("/home/darshan/Documents/DigitRecognizer/data/", "train.csv")
    train_set_x, train_set_y = load_data.load_data()

    print(train_set_y)



