import tensorlib as lib
from collections import namedtuple
import tensorflow as tf


class Block(namedtuple('Block', ['scope', 'unit_fn', 'args'])):
    pass


class WSConv2D(lib.layers.Conv2D):
    @lib.engine.add_arg_scope
    def __init__(self,
                 out_channels,
                 kernel_size,
                 strides=(1, 1),
                 padding='SAME',
                 data_format='channels_last',
                 activation=None,
                 use_bias=True,
                 epsilon=1e-5,
                 use_weight_standardization=False,
                 kernel_initializer='truncated_normal',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 dilation_rate=(1, 1),
                 **kwargs):
        super(WSConv2D, self).__init__(activation=activation,
                                       out_channels=out_channels,
                                       kernel_size=kernel_size,
                                       strides=strides,
                                       padding=padding,
                                       data_format=data_format,
                                       use_bias=use_bias,
                                       kernel_initializer=kernel_initializer,
                                       kernel_regularizer=kernel_regularizer,
                                       kernel_constraint=kernel_constraint,
                                       bias_initializer=bias_initializer,
                                       bias_regularizer=bias_regularizer,
                                       bias_constraint=bias_constraint,
                                       dilation_rate=dilation_rate,
                                       **kwargs)
        self.use_weight_standardization = use_weight_standardization
        self.epsilon = epsilon

    def build(self, input_shape):
        super(WSConv2D, self).build(input_shape)
        if self.use_weight_standardization:
            mean, var = tf.nn.moments(self.kernel, [0, 1, 2], keepdims=True)
            self.kernel = (self.kernel - mean) / tf.sqrt(var + self.epsilon)


@lib.engine.add_arg_scope
def ws_conv2d(
        inputs,
        out_channels,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding='SAME',
        rate=1,
        epsilon=1e-5,
        use_weight_standardization=False,
        kernel_initializer='truncated_normal',
        kernel_regularizer=None,
        bias_initializer='zeros',
        bias_regularizer=None,
        activation=None,
        normalizer_fn=None,
        normalizer_params=None,
        name=None):
    net = WSConv2D(
        out_channels=out_channels,
        kernel_size=kernel_size,
        strides=stride,
        padding=padding,
        dilation_rate=rate,
        epsilon=epsilon,
        use_weight_standardization=use_weight_standardization,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_initializer=bias_initializer,
        bias_regularizer=bias_regularizer,
        activation=None if normalizer_fn else activation,
        use_bias=not normalizer_fn and bias_initializer,
        name=name)
    if normalizer_fn is not None:
        if not issubclass(normalizer_fn, lib.engine.Layer):
            raise TypeError('Unexpected type of `normalizer_fn`!')
        normalizer_params = normalizer_params or {}
        outputs = net(inputs)
        outputs = normalizer_fn(name=net.name + '/batch_norm',
                                activation=activation,
                                **normalizer_params)(outputs)
    else:
        outputs = net(inputs)
    return outputs


def sub_sample(inputs, factor, name='sub_sample'):
    if factor == 1:
        return inputs
    else:
        return lib.layers.MaxPool2D(kernel_size=1, strides=factor, name=name)(inputs)


def conv2d_same(inputs, out_channels, kernel_size, stride, rate=1, name='conv_same'):
    if stride == 1:
        outputs = ws_conv2d(
            inputs, out_channels=out_channels,
            kernel_size=kernel_size, stride=stride, rate=rate,
            name=name)
    else:
        outputs = lib.layers.pad2d(
            inputs, kernel_size=kernel_size, rate=rate)
        outputs = ws_conv2d(
            outputs, out_channels=out_channels,
            kernel_size=kernel_size, stride=stride, rate=rate,
            name=name, padding='VALID')
    return outputs


def root_block_fn(net, depth_multiplier=1.0):
    net = conv2d_same(
        net,
        out_channels=int(64 * depth_multiplier),
        kernel_size=3,
        stride=2,
        name='conv1_1')
    net = conv2d_same(
        net,
        out_channels=int(64 * depth_multiplier),
        kernel_size=3,
        stride=1,
        name='conv1_2')
    net = conv2d_same(
        net,
        out_channels=int(128 * depth_multiplier),
        kernel_size=3,
        stride=1,
        name='conv1_3')
    return net


def stack_blocks_dense(net, blocks,
                       output_stride=None):
    current_stride = 4
    rate = 1
    for block in blocks:
        with lib.graph_scope(block.scope, 'block', [net]) as block_scope:
            net = block_scope.inputs
            block_stride = 1
            for i, unit in enumerate(block.args):
                point_name = 'unit_%d' % (i + 1)
                with lib.graph_scope(point_name, values=[net]) as unit_scope:
                    net = unit_scope.inputs
                    if output_stride is not None and current_stride == output_stride:
                        net = block.unit_fn(net, **dict(unit, stride=1))
                        rate *= unit.get('stride', 1)
                    else:
                        net = block.unit_fn(net, **unit)
                        current_stride *= unit.get('stride', 1)
                        if output_stride is not None and current_stride > output_stride:
                            raise ValueError("The target output_stride can not be reached.")
                    unit_scope.outputs = net
            if output_stride is not None and current_stride == output_stride:
                rate *= block_stride
            else:
                net = sub_sample(net, block_stride)
                current_stride *= block_stride
                if output_stride is not None and current_stride > output_stride:
                    raise ValueError("The target output_stride can not be reached.")
            block_scope.outputs = net
    if output_stride is not None and current_stride != output_stride:
        raise ValueError("The target output_stride can not be reached.")
    return net


def resnet_beta_arg_scope(
        weight_decay=0.0001,
        batch_norm_decay=0.997,
        batch_norm_epsilon=1e-5,
        activation='relu',
        use_batch_norm=True,
        use_weight_standardization=False):
    batch_norm_params = {
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
    }

    with lib.engine.arg_scope(
            [ws_conv2d],
            kernel_regularizer=lib.regularizers.l2(weight_decay),
            kernel_initializer='truncated_normal',
            activation=activation,
            normalizer_fn=lib.layers.BatchNorm if use_batch_norm else None,
            use_weight_standardization=use_weight_standardization):
        with lib.engine.arg_scope([lib.layers.BatchNorm], **batch_norm_params):
            with lib.engine.arg_scope([lib.layers.MaxPool2D], padding='SAME') as arg_sc:
                return arg_sc
