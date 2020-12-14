from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import tensor_shape
from tensorlib.engine import base_lib as F
from tensorlib.engine.base_layer import Layer, param_tracker
from tensorlib.utils.conv_util import *
from tensorlib import initializers
from tensorlib import regularizers
from tensorlib.utils import valid_value, check_mutex
from tensorlib.engine.scope_manager import add_arg_scope

__all__ = ['Conv1D', 'Conv2D', 'Conv3D', 'Conv',
           'SeparableConv2D', 'Conv2DTranspose',
           'DepthWiseConv2D', 'SeparableConv1D',
           'Pad', 'Pad1D', 'Pad2D', 'Pad3D',
           'pad1d', 'pad2d', 'pad3d', 'SeparableConv',
           'UpSampling1D', 'UpSampling2D', 'UpSampling3D']


class Conv(Layer):
    def __init__(self,
                 rank,
                 out_channels,
                 kernel_size,
                 strides=1,
                 padding='SAME',
                 data_format='channels_last',
                 activation=None,
                 use_bias=True,
                 kernel_initializer='truncated_normal',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 dilation_rate=1,
                 **kwargs):
        super(Conv, self).__init__(activation=activation, **kwargs)
        assert isinstance(rank, int) and 3 >= rank >= 1
        self.rank = rank
        self.out_channels = out_channels
        self.data_format = normalize_data_format(data_format, rank)
        self.kernel_size = normalize_tuple(kernel_size, rank, 'kernel size')
        self.strides = normalize_tuple(strides, rank, 'strides')
        self.dilation_rate = normalize_tuple(dilation_rate, rank, 'dilation_rate')
        self.padding = normalize_padding(padding)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = kernel_constraint
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_constraint = bias_constraint
        self._convolution_op = None
        self.kernel = None
        self.bias = None

    def build(self, input_shape):
        if len(input_shape) > self.rank + 2:
            raise ValueError("Input to {} should has {:d} rank,"
                             "but received input shape {}".format(
                              self.name, self.rank, str(input_shape)))
        channel_axis = -1 if self.data_format[-1] == 'C' else 1
        in_channels = input_shape[channel_axis]
        if not in_channels:
            raise ValueError("Input channel must be defined legally,"
                             "but received {}".format(str(in_channels)))
        kernel_shape = self.kernel_size + (in_channels, self.out_channels)
        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      name='kernel')
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.out_channels,),
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint,
                                        name='bias')
        else:
            self.bias = None
        self._convolution_op = nn_ops.Convolution(
            input_shape=tensor_shape.TensorShape(input_shape),
            filter_shape=self.kernel.shape,
            dilation_rate=self.dilation_rate,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format)

    def forward(self, inputs):
        outputs = self._convolution_op(inputs, self.kernel)
        if self.use_bias:
            if self.data_format[-1] != 'C':
                if self.rank == 1:
                    bias = array_ops.reshape(self.bias, (1, self.out_channels, 1))
                    outputs += bias
                else:
                    outputs = nn.bias_add(outputs, self.bias, data_format=self.data_format)
            else:
                outputs = nn.bias_add(outputs, self.bias, data_format=self.data_format)
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs


class Conv1D(Conv):
    @param_tracker()
    @add_arg_scope
    def __init__(self,
                 out_channels,
                 kernel_size,
                 strides=(1,),
                 padding='SAME',
                 data_format='channels_last',
                 activation=None,
                 use_bias=True,
                 kernel_initializer='truncated_normal',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 dilation_rate=(1,),
                 **kwargs):
        super(Conv1D, self).__init__(rank=1,
                                     activation=activation,
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


class Conv2D(Conv):
    @param_tracker()
    @add_arg_scope
    def __init__(self,
                 out_channels,
                 kernel_size,
                 strides=(1, 1),
                 padding='SAME',
                 data_format='channels_last',
                 activation=None,
                 use_bias=True,
                 kernel_initializer='truncated_normal',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 dilation_rate=(1, 1),
                 **kwargs):
        super(Conv2D, self).__init__(rank=2,
                                     activation=activation,
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


class Conv3D(Conv):
    @param_tracker()
    @add_arg_scope
    def __init__(self,
                 out_channels,
                 kernel_size,
                 strides=(1, 1, 1),
                 padding='SAME',
                 data_format='channels_last',
                 activation=None,
                 use_bias=True,
                 kernel_initializer='truncated_normal',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 dilation_rate=(1, 1, 1),
                 **kwargs):
        super(Conv3D, self).__init__(rank=3,
                                     activation=activation,
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


class Conv2DTranspose(Conv2D):
    @param_tracker()
    @add_arg_scope
    def __init__(self,
                 out_channels,
                 kernel_size,
                 spatial_size=None,
                 strides=(1, 1),
                 padding='SAME',
                 output_padding=None,
                 data_format='channels_last',
                 activation=None,
                 use_bias=True,
                 kernel_initializer='truncated_normal',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 dilation_rate=(1, 1),
                 **kwargs):
        super(Conv2DTranspose, self).__init__(activation=activation,
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
        if output_padding is not None:
            output_padding = normalize_tuple(output_padding, 2, 'output_padding')
            for stride, out_pad in zip(self.strides, self.output_padding):
                if out_pad >= stride:
                    raise ValueError('Stride ' + str(self.strides) + ' must be '
                                                                     ' greater than output padding ' +
                                     str(self.output_padding))
        else:
            output_padding = (None, None)
        self.output_padding = output_padding
        self.output_shape = None
        if spatial_size is not None:
            spatial_size = list(normalize_tuple(spatial_size, 2, 'spatial_size'))
        self.spatial_size = spatial_size

    def build(self, input_shape):
        if len(input_shape) != 4:
            raise ValueError("Inputs should have rank 4, but received:" + str(input_shape))
        if self.data_format[-1] == 'C':
            channel_axis = -1
            input_size = input_shape[1: -1]
        else:
            channel_axis = 1
            input_size = input_shape[2:]
        in_channels = input_shape[channel_axis]
        if not in_channels:
            raise ValueError("Input channel must be defined legally,"
                             "but received {}".format(str(in_channels)))
        if self.spatial_size is None:
            # Infer dynamic spatial size
            self.spatial_size = [
                deconv_output_length(
                    input_size[i], self.kernel_size[i],
                    self.strides[i], self.padding,
                    self.output_padding[i], self.dilation_rate[i])
                for i in range(len(input_size))]
        self.output_shape = (input_shape[0],) + tuple(self.spatial_size) + (in_channels,) \
            if channel_axis == -1 else (input_shape[0], in_channels) + tuple(self.spatial_size)
        kernel_shape = self.kernel_size + (self.out_channels, in_channels)
        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      name='kernel')
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.out_channels,),
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint,
                                        name='bias')
        else:
            self.bias = None

    def forward(self, inputs):
        outputs = F.conv2d_transpose(
            x=inputs,
            kernel=self.kernel,
            output_shape=self.output_shape,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate)
        if self.use_bias:
            outputs = nn.bias_add(
                value=outputs,
                bias=self.bias,
                data_format=self.data_format)
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs


class SeparableConv(Conv):
    def __init__(self,
                 rank,
                 out_channels,
                 kernel_size,
                 strides=1,
                 padding='SAME',
                 data_format='channels_last',
                 depth_multiplier=1,
                 depthwise_initializer='truncated_normal',
                 depthwise_regularizer=None,
                 pointwise_initializer='truncated_normal',
                 pointwise_regularizer=None,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 dilation_rate=1,
                 use_bias=True,
                 activation=None,
                 **kwargs):
        super(SeparableConv, self).__init__(
            rank=rank,
            out_channels=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            bias_initializer=bias_initializer,
            bias_regularizer=bias_regularizer,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            **kwargs)
        self.depth_multiplier = depth_multiplier
        self.depthwise_regularizer = initializers.get(depthwise_regularizer)
        self.depthwise_initializer = initializers.get(depthwise_initializer)
        self.pointwise_initializer = initializers.get(pointwise_initializer)
        self.pointwise_regularizer = regularizers.get(pointwise_regularizer)
        self.depthwise_kernel = None
        self.pointwise_kernel = None
        self.bias = None

    def build(self, input_shape):
        if len(input_shape) > self.rank + 2:
            raise ValueError("Input to {} should has {:d} rank,"
                             "but received input shape {}".format(
                              self.name, self.rank, str(input_shape)))
        if self.data_format[-1] == 'C':
            channel_axis = -1
        else:
            channel_axis = 1
        in_channels = input_shape[channel_axis]
        if not in_channels:
            raise ValueError("Input channel must be defined legally,"
                             "but received {}".format(str(in_channels)))
        depth_kernel_shape = self.kernel_size + (in_channels, self.depth_multiplier)
        point_kernel_shape = (1,) * self.rank + (self.depth_multiplier * in_channels, self.out_channels)
        self.depthwise_kernel = self.add_weight(shape=depth_kernel_shape,
                                                initializer=self.depthwise_initializer,
                                                regularizer=self.depthwise_regularizer,
                                                name='depthwise_kernel')
        self.pointwise_kernel = self.add_weight(shape=point_kernel_shape,
                                                initializer=self.pointwise_initializer,
                                                regularizer=self.pointwise_regularizer,
                                                name='pointwise_kernel')
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.out_channels,),
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        name='bias')
        self._built = True

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class SeparableConv1D(SeparableConv):
    @param_tracker()
    @add_arg_scope
    def __init__(self,
                 out_channels,
                 kernel_size,
                 strides=(1,),
                 padding='SAME',
                 data_format='channels_last',
                 depth_multiplier=1,
                 use_bias=False,
                 depthwise_initializer='truncated_normal',
                 depthwise_regularizer=None,
                 pointwise_initializer='truncated_normal',
                 pointwise_regularizer=None,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 dilation_rate=(1,),
                 activation=None,
                 **kwargs):
        super(SeparableConv1D, self).__init__(
            rank=1,
            out_channels=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            data_format=data_format,
            dilation_rate=dilation_rate,
            padding=padding,
            use_bias=use_bias,
            depth_multiplier=depth_multiplier,
            depthwise_initializer=depthwise_initializer,
            depthwise_regularizer=depthwise_regularizer,
            pointwise_initializer=pointwise_initializer,
            pointwise_regularizer=pointwise_regularizer,
            bias_initializer=bias_initializer,
            bias_regularizer=bias_regularizer,
            activation=activation,
            **kwargs)

    def forward(self, inputs):
        outputs = F.separable_conv1d(
            value=inputs,
            depthwise_kernel=self.depthwise_kernel,
            pointwise_kernel=self.pointwise_kernel,
            bias=self.bias if self.use_bias else None,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate)
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs


class SeparableConv2D(SeparableConv):
    @param_tracker()
    @add_arg_scope
    def __init__(self,
                 out_channels,
                 kernel_size,
                 strides=(1, 1),
                 padding='SAME',
                 data_format='channels_last',
                 depth_multiplier=1,
                 use_bias=False,
                 depthwise_initializer='truncated_normal',
                 depthwise_regularizer=None,
                 pointwise_initializer='truncated_normal',
                 pointwise_regularizer=None,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 **kwargs):
        super(SeparableConv2D, self).__init__(
            rank=2,
            out_channels=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            data_format=data_format,
            dilation_rate=dilation_rate,
            padding=padding,
            use_bias=use_bias,
            depth_multiplier=depth_multiplier,
            depthwise_initializer=depthwise_initializer,
            depthwise_regularizer=depthwise_regularizer,
            pointwise_initializer=pointwise_initializer,
            pointwise_regularizer=pointwise_regularizer,
            bias_initializer=bias_initializer,
            bias_regularizer=bias_regularizer,
            activation=activation,
            **kwargs)

    def forward(self, inputs):
        strides = (1,) + self.strides + (1,)\
            if self.data_format[-1] == 'C' else (1, 1) + self.strides
        outputs = nn.separable_conv2d(
            input=inputs,
            depthwise_filter=self.depthwise_kernel,
            pointwise_filter=self.pointwise_kernel,
            strides=strides,
            padding=self.padding,
            data_format=self.data_format,
            dilations=self.dilation_rate)
        if self.use_bias:
            outputs = nn.bias_add(outputs, self.bias,
                                  data_format=self.data_format)
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs


class DepthWiseConv2D(Conv2D):
    @param_tracker()
    @add_arg_scope
    def __init__(
            self,
            kernel_size,
            depth_multiplier=1,
            strides=(1, 1),
            padding='SAME',
            dilation_rate=(1, 1),
            use_bias=True,
            data_format='channels_last',
            depthwise_initializer='truncated_normal',
            depthwise_regularizer=None,
            bias_initializer='zeros',
            bias_regularizer=None,
            activation=None,
            **kwargs):
        super(DepthWiseConv2D, self).__init__(
            out_channels=None,
            kernel_size=kernel_size,
            data_format=data_format,
            strides=strides,
            padding=padding,
            dilation_rate=dilation_rate,
            use_bias=use_bias,
            bias_initializer=bias_initializer,
            bias_regularizer=bias_regularizer,
            activation=activation,
            **kwargs)
        self.depth_multiplier = depth_multiplier
        self.depthwise_initializer = initializers.get(depthwise_initializer)
        self.depthwise_regularizer = regularizers.get(depthwise_regularizer)

    def build(self, input_shape):
        if len(input_shape) < 4:
            raise ValueError('Inputs to `DepthwiseConv2D` should have rank 4. '
                             'Received input shape:', str(input_shape))
        channel_axis = self.data_format.find('C')
        in_channels = input_shape[channel_axis]
        if not in_channels:
            raise ValueError("Input channel must be defined legally,"
                             "but received {}".format(str(in_channels)))
        kernel_shape = self.kernel_size + (in_channels, self.depth_multiplier)
        self.kernel = self.add_weight(name="depthwise_kernel",
                                      shape=kernel_shape,
                                      initializer=self.depthwise_initializer,
                                      regularizer=self.depthwise_regularizer)
        if self.use_bias:
            self.bias = self.add_weight(name="bias",
                                        shape=(in_channels * self.depth_multiplier,),
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer)

    def forward(self, inputs):
        strides = (1,) + self.strides + (1,)\
            if self.data_format[-1] == 'C' else (1, 1) + self.strides
        outputs = nn.depthwise_conv2d(
            input=inputs,
            filter=self.kernel,
            strides=strides,
            padding=self.padding,
            data_format=self.data_format,
            rate=self.dilation_rate)
        if self.use_bias:
            outputs = F.bias_add(outputs, self.bias,
                                 data_format=self.data_format)
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs


class UpSampling1D(Layer):
    @param_tracker()
    def __init__(self,
                 size=2,
                 **kwargs):
        super(UpSampling1D, self).__init__(**kwargs)
        self.size = int(size)

    def forward(self, inputs):
        outputs = F.repeat_elements(inputs, self.size, axis=1)
        return outputs


class UpSampling2D(Layer):
    @param_tracker()
    def __init__(self,
                 size=(2, 2),
                 data_format='channels_last',
                 interpolation='nearest',
                 **kwargs):
        super(UpSampling2D, self).__init__(**kwargs)
        self.size = normalize_tuple(size, 2, 'size')
        self.data_format = normalize_data_format(data_format, 2)
        self.interpolation = valid_value(interpolation, ('nearest', 'bilinear'))

    def forward(self, inputs):
        outputs = F.resize_image(
            inputs, self.size[0], self.size[1],
            data_format=self.data_format,
            interpolation=self.interpolation)
        return outputs


class UpSampling3D(Layer):
    @param_tracker()
    def __init__(self,
                 size=(2, 2, 2),
                 data_format='channels_last',
                 **kwargs):
        super(UpSampling3D, self).__init__(**kwargs)
        self.size = normalize_tuple(size, 3, 'size')
        self.data_format = normalize_data_format(data_format, 3)

    def forward(self, inputs):
        outputs = F.resize_volumes(
            inputs, self.size,
            data_format=self.data_format)
        return outputs


class Pad(Layer):
    def __init__(self,
                 rank,
                 pad=None,
                 kernel_size=None,
                 rate=1,
                 mode='CONSTANT',
                 const_value=0,
                 data_format='channels_last',
                 activation=None,
                 **kwargs):
        super(Pad, self).__init__(activation=activation, **kwargs)
        assert isinstance(rank, int) and 3 >= rank >= 1
        self.rank = rank
        self.mode = valid_value(mode.upper(), ['CONSTANT', 'REFLECT', 'SYMMETRIC'])
        assert isinstance(const_value, int)
        self.const_value = const_value
        check_mutex(pad, kernel_size, names=['pad', 'kernel_size'])
        if pad is not None:
            self.pad = normalize_tuple(pad, rank, 'pad')
            padding = [[v] * 2 for v in self.pad]
        else:
            self.kernel_size = normalize_tuple(kernel_size, rank, 'kernel_size')
            self.rate = rate
            pad_total = [v + (v - 1) * (self.rate - 1) - 1
                         for v in self.kernel_size]
            padding = []
            for v in pad_total:
                pad_beg = v // 2
                pad_end = v - pad_beg
                padding.append([pad_beg, pad_end])
        self.data_format = normalize_data_format(data_format, rank)
        if self.data_format[-1] == 'C':
            self.padding = [[0, 0]] + padding + [[0, 0]]
        else:
            self.padding = [[0, 0], [0, 0]] + padding
        self._built = True

    def forward(self, inputs):
        outputs = array_ops.pad(inputs, self.padding, mode=self.mode,
                                constant_values=self.const_value)
        if self.activation:
            outputs = self.activation(outputs)
        return outputs


class Pad1D(Pad):
    @param_tracker()
    @add_arg_scope
    def __init__(self,
                 pad=None,
                 kernel_size=None,
                 rate=1,
                 mode='CONSTANT',
                 const_value=0,
                 data_format='channels_last',
                 activation=None,
                 **kwargs):
        super(Pad1D, self).__init__(
            1, pad=pad, kernel_size=kernel_size,
            rate=rate, mode=mode, const_value=const_value,
            data_format=data_format, activation=activation, **kwargs)


class Pad2D(Pad):
    @param_tracker()
    @add_arg_scope
    def __init__(self,
                 pad=None,
                 kernel_size=None,
                 rate=1,
                 mode='CONSTANT',
                 const_value=0,
                 data_format='channels_last',
                 activation=None,
                 **kwargs):
        super(Pad2D, self).__init__(
            2, pad=pad, kernel_size=kernel_size,
            rate=rate, mode=mode, const_value=const_value,
            data_format=data_format, activation=activation, **kwargs)


class Pad3D(Pad):
    @param_tracker()
    @add_arg_scope
    def __init__(self,
                 pad=None,
                 kernel_size=None,
                 rate=1,
                 mode='CONSTANT',
                 const_value=0,
                 data_format='channels_last',
                 activation=None,
                 **kwargs):
        super(Pad3D, self).__init__(
            3, pad=pad, kernel_size=kernel_size,
            rate=rate, mode=mode, const_value=const_value,
            data_format=data_format, activation=activation, **kwargs)


def pad1d(inputs,
          pad=None,
          kernel_size=None,
          rate=1,
          mode='CONSTANT',
          data_format='channels_last',
          const_value=0,
          **kwargs):
    return Pad1D(pad=pad,
                 kernel_size=kernel_size,
                 rate=rate,
                 mode=mode,
                 const_value=const_value,
                 data_format=data_format,
                 **kwargs)(inputs)


def pad2d(inputs,
          pad=None,
          kernel_size=None,
          rate=1,
          mode='CONSTANT',
          data_format='channels_last',
          const_value=0,
          **kwargs):
    return Pad2D(pad=pad,
                 kernel_size=kernel_size,
                 rate=rate,
                 mode=mode,
                 const_value=const_value,
                 data_format=data_format,
                 **kwargs)(inputs)


def pad3d(inputs,
          pad=None,
          kernel_size=None,
          rate=1,
          mode='CONSTANT',
          data_format='channels_last',
          const_value=0,
          **kwargs):
    return Pad3D(pad=pad,
                 kernel_size=kernel_size,
                 rate=rate,
                 mode=mode,
                 const_value=const_value,
                 data_format=data_format,
                 **kwargs)(inputs)
