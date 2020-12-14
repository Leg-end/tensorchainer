import tensorlib as lib
from collections import namedtuple


class Block(namedtuple('Block', ['name', 'unit', 'args'])):
    pass


def sub_sample(factor, name='sub_sample'):
    if factor == 1:
        def linear(inputs): return inputs
        return linear
    else:
        return lib.layers.MaxPool2D(kernel_size=1, strides=factor, name=name)


def conv2d_same(out_channels, kernel_size, stride, rate=1, name='conv_same'):
    if stride == 1:
        return lib.contrib.WSConv2D(
            out_channels=out_channels,
            kernel_size=kernel_size,
            strides=stride, rate=rate,
            name=name)
    else:
        return lib.Sequential(
            lib.layers.Pad2D(
                kernel_size=kernel_size, rate=rate),
            lib.contrib.WSConv2D(
                out_channels=out_channels,
                kernel_size=kernel_size,
                strides=stride, rate=rate,
                name=name, padding='VALID'),
            name='')


def root_block(depth_multiplier=1.0):
    return lib.Sequential(conv2d_same(
        out_channels=int(64 * depth_multiplier),
        kernel_size=3, stride=2,
        name='conv1_1'), conv2d_same(
        out_channels=int(64 * depth_multiplier),
        kernel_size=3, stride=1,
        name='conv1_2'), conv2d_same(
        out_channels=int(128 * depth_multiplier),
        kernel_size=3, stride=1,
        name='conv1_3'), name='')


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
            [lib.contrib.WSConv2D],
            kernel_regularizer=lib.regularizers.l2(weight_decay),
            kernel_initializer='truncated_normal',
            activation=activation,
            normalizer_fn=lib.layers.BatchNorm if use_batch_norm else None,
            use_weight_standardization=use_weight_standardization):
        with lib.engine.arg_scope([lib.layers.BatchNorm], **batch_norm_params):
            with lib.engine.arg_scope([lib.layers.MaxPool2D], padding='SAME') as arg_sc:
                return arg_sc
