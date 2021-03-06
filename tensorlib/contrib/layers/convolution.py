import tensorlib as lib
from tensorflow.python.ops import nn
from tensorflow.python.ops import math_ops


class _WSConv2D(lib.layers.Conv2D):
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
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 dilation_rate=(1, 1),
                 **kwargs):
        super(_WSConv2D, self).__init__(activation=activation,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,
                                        strides=strides,
                                        padding=padding,
                                        data_format=data_format,
                                        use_bias=use_bias,
                                        kernel_initializer=kernel_initializer,
                                        kernel_regularizer=kernel_regularizer,
                                        bias_initializer=bias_initializer,
                                        bias_regularizer=bias_regularizer,
                                        dilation_rate=dilation_rate,
                                        **kwargs)
        self.use_weight_standardization = use_weight_standardization
        self.epsilon = epsilon

    def build(self, input_shape):
        super(_WSConv2D, self).build(input_shape)
        if self.use_weight_standardization:
            mean, var = nn.moments(self.kernel, [0, 1, 2], keepdims=True)
            self.kernel = (self.kernel - mean) / math_ops.sqrt(var + self.epsilon)


def Convolution(out_channels,
                kernel_size,
                rank,
                strides=1,
                rate=1,
                padding='SAME',
                data_format='channels_last',
                kernel_initializer='truncated_normal',
                kernel_regularizer=None,
                bias_initializer='zeros',
                bias_regularizer=None,
                activation=None,
                normalizer=None,
                normalizer_params=None,
                trainable=False,
                name=None):
    if rank == 1:
        layer_class = lib.layers.Conv1D
    elif rank == 2:
        layer_class = lib.layers.Conv2D
    elif rank == 3:
        layer_class = lib.layers.Conv3D
    else:
        raise ValueError("Rank must be in [1, 2, 3], but received:", rank)
    conv = layer_class(
        out_channels=out_channels,
        kernel_size=kernel_size,
        strides=strides,
        dilation_rate=rate,
        data_format=data_format,
        padding=padding,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_initializer=bias_initializer,
        bias_regularizer=bias_regularizer,
        use_bias=not normalizer and bias_initializer,
        activation=None if normalizer else activation,
        trainable=trainable,
        name=name)
    if normalizer is not None:
        assert issubclass(normalizer, lib.Layer)
        normalizer_params = normalizer_params or {}
        bn = normalizer(name=conv.name + '/batch_norm',
                        activation=activation,
                        **normalizer_params)
        return lib.Sequential(conv, bn, name='')
    return conv


@lib.engine.add_arg_scope
def WSConv2D(out_channels,
             kernel_size,
             strides=(1, 1),
             padding='SAME',
             rate=1,
             data_format='channels_last',
             epsilon=1e-5,
             use_weight_standardization=False,
             kernel_initializer='truncated_normal',
             kernel_regularizer=None,
             bias_initializer='zeros',
             bias_regularizer=None,
             activation=None,
             normalizer=None,
             normalizer_params=None,
             trainable=False,
             name=None):
    conv = _WSConv2D(
        out_channels=out_channels,
        kernel_size=kernel_size,
        strides=strides,
        dilation_rate=rate,
        data_format=data_format,
        padding=padding,
        epsilon=epsilon,
        use_weight_standardization=use_weight_standardization,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_initializer=bias_initializer,
        bias_regularizer=bias_regularizer,
        use_bias=not normalizer and bias_initializer,
        activation=None if normalizer else activation,
        trainable=trainable,
        name=name)
    if normalizer is not None:
        assert issubclass(normalizer, lib.Layer)
        normalizer_params = normalizer_params or {}
        bn = normalizer(name=conv.name + '/batch_norm',
                        activation=activation,
                        **normalizer_params)
        return lib.Sequential(conv, bn, name='')
    return conv


@lib.engine.add_arg_scope
def SeparableConv2D(out_channels,
                    kernel_size,
                    strides=(1, 1),
                    rate=(1, 1),
                    padding='SAME',
                    data_format='channels_last',
                    depth_multiplier=1,
                    kernel_initializer='truncated_normal',
                    kernel_regularizer=None,
                    pointwise_initializer='truncated_normal',
                    pointwise_regularizer=None,
                    bias_initializer='zeros',
                    bias_regularizer=None,
                    activation=None,
                    normalizer=None,
                    normalizer_params=None,
                    trainable=False,
                    name=None):
    if pointwise_initializer is None:
        pointwise_initializer = kernel_initializer
    if out_channels is not None:
        conv = lib.layers.SeparableConv2D(
            out_channels=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            dilation_rate=rate,
            data_format=data_format,
            padding=padding,
            depth_multiplier=depth_multiplier,
            depthwise_initializer=kernel_initializer,
            depthwise_regularizer=kernel_regularizer,
            pointwise_initializer=pointwise_initializer,
            pointwise_regularizer=pointwise_regularizer,
            bias_initializer=bias_initializer,
            bias_regularizer=bias_regularizer,
            use_bias=not normalizer and bias_initializer,
            activation=None if normalizer else activation,
            trainable=trainable,
            name=name)
    else:
        conv = lib.layers.DepthWiseConv2D(
            kernel_size=kernel_size,
            strides=strides,
            dilation_rate=rate,
            data_format=data_format,
            padding=padding,
            depth_multiplier=depth_multiplier,
            depthwise_initializer=kernel_initializer,
            depthwise_regularizer=kernel_regularizer,
            bias_initializer=bias_initializer,
            bias_regularizer=bias_regularizer,
            use_bias=not normalizer and bias_initializer,
            activation=None if normalizer else activation,
            trainable=trainable,
            name=name)
    if normalizer is not None:
        assert issubclass(normalizer, lib.Layer)
        normalizer_params = normalizer_params or {}
        bn = normalizer(name=conv.name + '/batch_norm',
                        activation=activation,
                        **normalizer_params)
        return lib.Sequential(conv, bn, name='')
    return conv


@lib.engine.add_arg_scope
def Conv1D(out_channels,
           kernel_size,
           strides=(1,),
           rate=(1,),
           padding='SAME',
           data_format='channels_last',
           kernel_initializer='truncated_normal',
           kernel_regularizer=None,
           bias_initializer='zeros',
           bias_regularizer=None,
           activation=None,
           normalizer=None,
           normalizer_params=None,
           trainable=False,
           name=None):
    return Convolution(rank=1,
                       out_channels=out_channels,
                       kernel_size=kernel_size,
                       strides=strides,
                       rate=rate,
                       padding=padding,
                       data_format=data_format,
                       kernel_initializer=kernel_initializer,
                       kernel_regularizer=kernel_regularizer,
                       bias_initializer=bias_initializer,
                       bias_regularizer=bias_regularizer,
                       activation=activation,
                       normalizer=normalizer,
                       normalizer_params=normalizer_params,
                       trainable=trainable,
                       name=name)


@lib.engine.add_arg_scope
def Conv2D(out_channels,
           kernel_size,
           strides=(1, 1),
           rate=(1, 1),
           padding='SAME',
           data_format='channels_last',
           kernel_initializer='truncated_normal',
           kernel_regularizer=None,
           bias_initializer='zeros',
           bias_regularizer=None,
           activation=None,
           normalizer=None,
           normalizer_params=None,
           trainable=False,
           name=None):
    return Convolution(rank=2,
                       out_channels=out_channels,
                       kernel_size=kernel_size,
                       strides=strides,
                       rate=rate,
                       padding=padding,
                       data_format=data_format,
                       kernel_initializer=kernel_initializer,
                       kernel_regularizer=kernel_regularizer,
                       bias_initializer=bias_initializer,
                       bias_regularizer=bias_regularizer,
                       activation=activation,
                       normalizer=normalizer,
                       normalizer_params=normalizer_params,
                       trainable=trainable,
                       name=name)


@lib.engine.add_arg_scope
def Conv3D(out_channels,
           kernel_size,
           strides=(1, 1, 1),
           rate=(1, 1, 1),
           padding='SAME',
           data_format='channels_last',
           kernel_initializer='truncated_normal',
           kernel_regularizer=None,
           bias_initializer='zeros',
           bias_regularizer=None,
           activation=None,
           normalizer=None,
           normalizer_params=None,
           trainable=False,
           name=None):
    return Convolution(rank=3,
                       out_channels=out_channels,
                       kernel_size=kernel_size,
                       strides=strides,
                       rate=rate,
                       padding=padding,
                       data_format=data_format,
                       kernel_initializer=kernel_initializer,
                       kernel_regularizer=kernel_regularizer,
                       bias_initializer=bias_initializer,
                       bias_regularizer=bias_regularizer,
                       activation=activation,
                       normalizer=normalizer,
                       normalizer_params=normalizer_params,
                       trainable=trainable,
                       name=name)
