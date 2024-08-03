import tensorflow as tf
from tensorflow.python.training import moving_averages

slim = tf.contrib.slim


def conv2d(inputs,
           filter_height,
           filter_width,
           output_channels,
           stride=(1, 1),
           padding='SAME',
           name='Conv2D'):
    input_channels = int(inputs.get_shape()[-1])
    fan_in = filter_height * filter_width * input_channels
    stddev = tf.sqrt(2.0 / fan_in)
    weights_shape = [
        filter_height, filter_width, input_channels, output_channels
    ]
    biases_shape = [output_channels]

    with tf.variable_scope(name):
        #变量的初始化方式设置为正态分布
        filters_init = tf.truncated_normal_initializer(stddev=stddev)
        #使用常量0.1来初始化
        biases_init = tf.constant_initializer(0)

        filters = tf.get_variable(
            'weights',
            shape=weights_shape,
            initializer=filters_init,
            collections=['weights', 'variables'])
        biases = tf.get_variable(
            'biases',
            shape=biases_shape,
            initializer=biases_init,
            collections=['biases', 'variables'])
        return tf.nn.conv2d(
            inputs, filters, strides=[1, *stride, 1], padding=padding) + biases


def conv2dplus(inputs,
               filter_height,
               filter_width,
               output_channels,
               stride=(1, 1),
               gaussian=0,
               pad=0,
               padding='SAME',
               name='Conv2Dp'):
    #inputs = tf.pad(input,pad)
    input_channels = int(inputs.get_shape()[-1])
    fan_in = filter_height * filter_width * input_channels
    #是否手動設置正态分布值
    if gaussian == 0:
        stddev = tf.sqrt(2.0 / fan_in)
    else:
        stddev = gaussian

    weights_shape = [
        filter_height, filter_width, input_channels, output_channels
    ]
    biases_shape = [output_channels]

    with tf.variable_scope(name):
        #变量的初始化方式设置为正态分布
        filters_init = tf.truncated_normal_initializer(stddev=stddev)
        #使用常量0来初始化
        biases_init = tf.constant_initializer(0)

        filters = tf.get_variable(
            'weights',
            shape=weights_shape,
            initializer=filters_init,
            collections=['weights', 'variables'])
        biases = tf.get_variable(
            'biases',
            shape=biases_shape,
            initializer=biases_init,
            collections=['biases', 'variables'])
        return tf.nn.conv2d(
            inputs, filters, strides=[1, *stride, 1], padding=padding) + biases


def deconv2d(inputs,
             filter_height,
             filter_width,
             output_shape,
             stride=(1, 1),
             padding='SAME',
             name='Deconv2D'):
    input_channels = int(inputs.get_shape()[-1])
    #反
    output_channels = output_shape[-1]
    fan_in = filter_height * filter_width * output_channels
    stddev = tf.sqrt(2.0 / fan_in)
    weights_shape = [
        filter_height, filter_width, output_channels, input_channels
    ]
    biases_shape = [output_channels]

    with tf.variable_scope(name):
        filters_init = tf.truncated_normal_initializer(stddev=stddev)
        biases_init = tf.constant_initializer(0.1)

        filters = tf.get_variable(
            'weights',
            shape=weights_shape,
            initializer=filters_init,
            collections=['weights', 'variables'])
        biases = tf.get_variable(
            'biases',
            shape=biases_shape,
            initializer=biases_init,
            collections=['biases', 'variables'])
        return tf.nn.conv2d_transpose(
            inputs,
            filters,
            output_shape,
            strides=[1, *stride, 1],
            padding=padding) + biases


def deconv2dplus(inputs,
                 filter_height,
                 filter_width,
                 output_shape,
                 stride=(1, 1),
                 pad=0,
                 padding='SAME',
                 name='Deconv2Dp'):
    #inputs = tf.pad(input,pad)
    input_channels = int(inputs.get_shape()[-1])
    #反
    output_channels = output_shape[-1]
    fan_in = filter_height * filter_width * output_channels
    stddev = tf.sqrt(2.0 / fan_in)
    weights_shape = [
        filter_height, filter_width, output_channels, input_channels
    ]
    biases_shape = [output_channels]

    with tf.variable_scope(name):
        filters_init = tf.truncated_normal_initializer(stddev=stddev)
        biases_init = tf.constant_initializer(0.1)

        filters = tf.get_variable(
            'weights',
            shape=weights_shape,
            initializer=filters_init,
            collections=['weights', 'variables'])
        biases = tf.get_variable(
            'biases',
            shape=biases_shape,
            initializer=biases_init,
            collections=['biases', 'variables'])
        return tf.nn.conv2d_transpose(
            inputs,
            filters,
            output_shape,
            strides=[1, *stride, 1],
            padding=padding) + biases


def relu(inputs, name='Relu'):
    return tf.nn.relu(inputs, name)


def leaky_relu(inputs, leak=0.1, name='LeakyRelu'):
    with tf.name_scope(name):
        return tf.maximum(inputs, leak * inputs)


def batch_norm(inputs, name='Batch_Norm'):
    with tf.variable_scope(name):
        x_shape = inputs.get_shape()
        params_shape = x_shape[-1:]

        axis = list(range(len(x_shape)))

        beta = tf.get_variable(
            'beta',
            shape=params_shape,
            initializer=tf.zeros_initializer,
            collections=['beta', 'variables'])
        gama = tf.get_variable(
            'gama',
            shape=params_shape,
            initializer=tf.ones_initializer,
            collections=['gama', 'variables'])
        moving_mean = tf.get_variable(
            'moving_mean',
            shape=params_shape,
            initializer=tf.zeros_initializer,
            collections=['moving_mean', 'variables'])
        moving_variance = tf.get_variable(
            'moving_variance',
            shape=params_shape,
            initializer=tf.ones_initializer,
            collections=['moving_variance', 'variables'])

        mean, variance = tf.nn.moments(inputs, axis)
        update_moving_mean = moving_averages.assign_moving_average(
            moving_mean, mean, 0.9997)
        update_moving_variance = moving_averages.assign_moving_average(
            moving_variance, variance, 0.9997)

        return tf.nn.batch_normalization(inputs, mean, variance, beta, gama,
                                         0.001)


def para_relu(inputs, leak=0.1, name='PRelu'):
    with tf.name_scope(name):
        return tf.maximum(inputs, 0.0) + tf.minimum(0.0, leak * inputs)


def _get_variable(name, shape, initializer, weight_decay=0.0, dtype='float'):
    if weight_decay > 0:
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    else:
        regularizer = None
    collections = [name, 'variables']
    return tf.get_variable(
        name,
        shape=shape,
        initializer=initializer,
        dtype=dtype,
        regularizer=regularizer,
        collections=collections)


def skipblock(inputs,
              filter_height,
              filter_width,
              output_channels,
              stride=(1, 1),
              gaussian=0,
              pad=0,
              padding='SAME',
              name='Skip_Block'):
    with tf.variable_scope(name):
        conv = conv2dplus(
            inputs,
            filter_height,
            filter_width,
            output_channels,
            pad=1,
            padding='SAME',
            gaussian=0.1179,
            name='conv')
        prelu = para_relu(conv, name='prelu')
        batchnorm = batch_norm(prelu, name='Batch_Norm')
        return batchnorm
