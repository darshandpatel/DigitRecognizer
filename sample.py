from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs
from theano.sandbox.cuda.basic_ops import gpu_contiguous
import theano.tensor as T
from theano.tensor.nnet import conv
import numpy

filters_data = T.fmatrix('filters_data')

input_data = numpy.asarray([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1],
[1, 1, 1, 1]])

filters_data = numpy.asarray([[1, ], [1, 1]])

input_data = T.as_tensor_variable(input_data, name='input_data', ndim=2)

filters_data = T.as_tensor_variable(filters_data, name='filters_data', ndim=2)
conv_op = FilterActs(partial_sum=1)
contiguous_input = gpu_contiguous(input_data)
contiguous_filters = gpu_contiguous(filters_data)

out = conv_op(contiguous_input, contiguous_filters)