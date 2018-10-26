import tensorflow as tf
import numpy as np

def weight_variable(shape, stddev=0.01):
    """
    Initialize a weight variable with given shape,
    by sampling randomly from Normal(0.0, stddev^2).
    :param shape: list(int).
    :param stddev: float, standard deviation of Normal distribution for weights.
    :return weights: tf.Variable.
    """
    weights = tf.get_variable('weights', shape, tf.float32,
                              tf.random_normal_initializer(mean=0.0, stddev=stddev))
    return weights


def bias_variable(shape, value=1.0):
    """
    Initialize a bias variable with given shape,
    with given constant value.
    :param shape: list(int).
    :param value: float, initial value for biases.
    :return biases: tf.Variable.
    """
    biases = tf.get_variable('biases', shape, tf.float32,
                             tf.constant_initializer(value=value))
    return biases


def conv2d(x, W, stride, padding='SAME'):
    """
    Compute a 2D convolution from given input and filter weights.
    :param x: tf.Tensor, shape: (N, H, W, C).
    :param W: tf.Tensor, shape: (fh, fw, ic, oc).
    :param stride: int, the stride of the sliding window for each dimension.
    :param padding: str, either 'SAME' or 'VALID',
                         the type of padding algorithm to use.
    :return: tf.Tensor.
    """
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)

def conv_layer_nn(x, side_l, stride, out_depth, padding='SAME', use_bias=True, **kwargs):
    """
    Add a new convolutional layer.
    :param x: tf.Tensor, shape: (N, H, W, C).
    :param side_l: int, the side length of the filters for each dimension.
    :param stride: int, the stride of the filters for each dimension.
    :param out_depth: int, the total number of filters to be applied.
    :param padding: str, either 'SAME' or 'VALID',
                         the type of padding algorithm to use.
    :param use_bias: bool, if True, bias use, else do not use bias,
                           after conv, typically use batchnormalization
    :param kwargs: dict, extra arguments, including weights/biases initialization hyperparameters.
        - weight_stddev: float, standard deviation of Normal distribution for weights.
        - biases_value: float, initial value for biases.
    :return: tf.Tensor.
    """
    weights_stddev = kwargs.pop('weights_stddev', 0.01)
    in_depth = int(x.get_shape()[-1])
    filters = weight_variable([side_l, side_l, in_depth, out_depth], stddev=weights_stddev)
    if use_bias:
        biases_value = kwargs.pop('biases_value', 0.1)
        biases = bias_variable([out_depth], value=biases_value)
        return conv2d(x, filters, stride, padding=padding) + biases
    else:
        return conv2d(x, filters, stride, padding=padding)
# def conv_layer(x, filters, kernel_size, strides, padding='SAME', use_bias=True):
#     return tf.layers.conv2d(x, filters, kernel_size, strides, padding, use_bias=use_bias)
def conv_layer(x, filters, kernel_size, strides, padding='SAME', use_bias=True):
    side_l = kernel_size[0]
    out_depth = filters
    stride = strides[0]
    conv = conv_layer_nn(x, side_l, stride, out_depth, padding=padding, use_bias=use_bias)
    return conv

def conv_bn_relu(x, filters, kernel_size, is_train, strides=(1, 1), padding='SAME', relu=True):
    """
    Add conv + bn + Relu layers.
    see conv_layer and batchNormalization function
    """
    conv = conv_layer(x, filters, kernel_size, strides, padding, use_bias=False)
    bn = batchNormalization(conv, is_train)
    if relu:
        return tf.nn.leaky_relu(bn, alpha=0.1)
    else:
        return bn

def max_pool(x, side_l, stride, padding='SAME'):
    """
    Performs max pooling on given input.
    :param x: tf.Tensor, shape: (N, H, W, C).
    :param side_l: int, the side length of the pooling window for each dimension.
    :param stride: int, the stride of the sliding window for each dimension.
    :param padding: str, either 'SAME' or 'VALID',
                         the type of padding algorithm to use.
    :return: tf.Tensor.
    """
    return tf.nn.max_pool(x, ksize=[1, side_l, side_l, 1],
                          strides=[1, stride, stride, 1], padding=padding)

def batchNormalization(x, is_train):
    """
    Add a new batchNormalization layer.
    :param x: tf.Tensor, shape: (N, H, W, C) or (N, D)
    :param is_train: tf.placeholder(bool), if True, train mode, else, test mode
    :return: tf.Tensor.
    """
    return tf.layers.batch_normalization(x, training=is_train, momentum=0.99, epsilon=0.001, center=True, scale=True)