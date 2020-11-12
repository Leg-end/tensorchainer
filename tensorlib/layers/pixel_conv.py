from tensorlib.layers.convolution import Conv2D
from tensorlib.engine.scope_manager import add_arg_scope
from tensorflow.python.framework.ops import convert_to_tensor
from tensorflow.python.ops.array_ops import split
from tensorlib.activation_ops import tanh, sigmoid
import numpy as np

__all__ = ["MaskConv2D", "GatedMaskConv2D"]

"""
    Blind spot in the receptive field be mentioned in paper:
    example:  more detail see arXiv:1606.05328v2  `Conditional Image Generation with PixelCNN Decoders` Figure1 Right
    >>>
        import numpy as np
        import tensorflow as tf
        np.set_printoptions(suppress=True)
        tf.enable_eager_execution()
        inputs_1 = np.arange(25).reshape([1, 5, 5, 1]).astype("f")
        inputs_1[:, 1, 4, :] = 5555.
        inputs_1[:, 2, 3, :] = 2222.
        inputs_1[:, 2, 4, :] = 1111.
        inputs_1 = tf.convert_to_tensor(inputs_1, dtype=tf.float32)
        kernel = np.ones([3, 3])
        kernel[2:, :] = 0.0
        kernel[1, 1:] = 0.0
        print("kernel \n", kernel)
        kernel = kernel.reshape((3, 3, 1, 1))
        kernel = tf.constant_initializer(value=kernel)
        cv_1 = Conv2D(out_channels=1, in_channels=1, kernel_size=(3, 3),
                      padding="VALID",
                      kernel_initializer=kernel, use_bias=False)
        cv_2 = Conv2D(out_channels=1, in_channels=1, kernel_size=(3, 3),
                      padding="VALID",
                      kernel_initializer=kernel, use_bias=False)
        m_1 = cv_1(inputs_1)
        out_1 = cv_2(m_1)
        out_1 = np.asarray(out_1, dtype=np.float32).transpose([0, 3, 1, 2])
        print("input_1 \n", tf.transpose(inputs_1, [0, 3, 1, 2]).numpy())
        print("m_1 \n", tf.transpose(m_1, [0, 3, 1, 2]).numpy())
        print("out_1 \n", out_1)
        print("........................")
        inputs_2 = np.arange(25).reshape([1, 5, 5, 1]).astype("f")
        inputs_2[:, 1, 4, :] = 500.
        inputs_2[:, 2, 3, :] = 200.
        inputs_2[:, 2, 4, :] = 100.
        inputs_2 = tf.convert_to_tensor(inputs_2, dtype=tf.float32)

        m_2 = cv_1(inputs_2)
        out_2 = cv_2(m_2)
        out_2 = np.asarray(out_2, dtype=np.float32).transpose([0, 3, 1, 2])
        print("input_2 \n ", tf.transpose(inputs_2, [0, 3, 1, 2]).numpy())
        print("m_2 \n", tf.transpose(m_2, [0, 3, 1, 2]).numpy())
        print("out_2 \n ", out_2)
    >>>
        kernel 
         [[1. 1. 1.]
         [1. 0. 0.]
         [0. 0. 0.]]
        input_1 
         [[[[   0.    1.    2.    3.    4.]
           [   5.    6.    7.    8. 5555.]
           [  10.   11.   12. 2222. 1111.]
           [  15.   16.   17.   18.   19.]
           [  20.   21.   22.   23.   24.]]]]
        m_1 
         [[[[   8.   12.   16.]
           [  28.   32. 5582.]
           [  48. 2261. 3362.]]]]
        out_1 
         [[[[64.]]]]
        ........................
        input_2 
          [[[[  0.   1.   2.   3.   4.]
           [  5.   6.   7.   8. 500.]
           [ 10.  11.  12. 200. 100.]
           [ 15.  16.  17.  18.  19.]
           [ 20.  21.  22.  23.  24.]]]]
        m_2 
         [[[[  8.  12.  16.]
           [ 28.  32. 527.]
           [ 48. 239. 329.]]]]
        out_2 
          [[[[64.]]]]
"""


class MaskConv2D(Conv2D):
    """
    Arg:
        mask_type: a string, either `B` or 'A'.

        more detail see: arXiv:1601.06759v3 `Pixel Recurrent Neural Networks` sec 3.4
    """

    @add_arg_scope
    def __init__(self,
                 out_channels,
                 kernel_size,
                 mask_type="B",
                 stack_type="v",
                 channel_split=None,
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
                 dilation_rate=1,
                 **kwargs):
        super(MaskConv2D, self).__init__(
            out_channels=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            kernel_constraint=kernel_constraint,
            bias_initializer=bias_initializer,
            bias_regularizer=bias_regularizer,
            bias_constraint=bias_constraint,
            dilation_rate=dilation_rate,
            **kwargs)

        if mask_type not in ["B", "A"]:
            raise ValueError("Got the unknown value %s for mask_type" % mask_type)
        self.mask_type = mask_type
        if stack_type not in ["v", "h"]:
            raise ValueError("Got the unknown value %s for stack_type" % stack_type)
        self.stack_type = stack_type
        self.channel_split = channel_split

    def build(self, input_shape):
        super(MaskConv2D, self).build(input_shape)
        kernel_shape = self.kernel.get_shape().as_list()
        mask = _mask_kernel(kernel_shape=kernel_shape,
                            mask_type=self.mask_type,
                            stack_type=self.stack_type,
                            channel_split=self.channel_split)
        self.kernel = self.kernel * convert_to_tensor(mask, dtype=self.kernel.dtype)


def _mask_kernel(kernel_shape,
                 mask_type="A",
                 stack_type="v",
                 channel_split=None):
    """
    :param kernel_shape:
        (kernel_height, kernel_width, input_channels, output_channels)
    :param channel_split: a string or None
        when channel_split is string, it's either "cross" or ""
    :return:
    """
    kernel_h, kernel_w, out_channels, in_channels = kernel_shape
    h_center, w_center = kernel_h // 2, kernel_w // 2
    if channel_split is not None:
        if in_channels // 3 == 0:
            raise RuntimeError(
                "When channel split is specified, the input channel must be at least >= 3")
        if out_channels // 3 == 0:
            raise RuntimeError(
                "When channel split is specified, the output channel must be at least >= 3")
        if channel_split == "cross":
            # [0, 1, 2, 0, ...] -> [R, G, B, R, ...]
            code_matrix = np.expand_dims(np.arange(out_channels) % 3, axis=0) - \
                          np.expand_dims(np.arange(in_channels) % 3, axis=1)
        else:
            raise ValueError("Got the unknown value %s for channel_spilt" % channel_split)
    else:
        if in_channels != out_channels:
            raise ValueError("When channel split is None, the output channels and input channels must be equal")
        code_matrix = np.expand_dims(np.arange(out_channels), axis=0) - np.expand_dims(np.arange(in_channels), axis=1)
    if mask_type == "A":
        in_index, out_index = np.where(code_matrix > 0)
    else:
        in_index, out_index = np.where(code_matrix >= 0)

    mask = np.ones(shape=kernel_shape)
    if stack_type == "v":
        mask[h_center:, :, :, :] = 0.0
        mask[h_center, :, out_index, in_index] = 1.0
    else:
        mask[:, w_center:, :, :] = 0.0
        mask[:, w_center, out_index, in_index] = 1.0
    return mask


class GatedMaskConv2D(MaskConv2D):
    @add_arg_scope
    def __init__(self,
                 out_channels,
                 kernel_size,
                 mask_type="B",
                 stack_type="v",
                 channel_split=None,
                 strides=(1, 1),
                 padding='SAME',
                 data_format='channels_last',
                 use_bias=True,
                 kernel_initializer='truncated_normal',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 dilation_rate=1,
                 **kwargs):
        super(GatedMaskConv2D, self).__init__(
            out_channels=out_channels * 2,
            kernel_size=kernel_size,
            mask_type=mask_type,
            stack_type=stack_type,
            channel_split=channel_split,
            strides=strides,
            padding=padding,
            data_format=data_format,
            activation=None,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            kernel_constraint=kernel_constraint,
            bias_initializer=bias_initializer,
            bias_regularizer=bias_regularizer,
            bias_constraint=bias_constraint,
            dilation_rate=dilation_rate,
            **kwargs)

    def forward(self, inputs):
        outputs = super(GatedMaskConv2D, self).forward(inputs=inputs)
        tanh_inputs, sigmoid_inputs = split(
            value=outputs, num_or_size_splits=2, axis=-1 if self.data_format[-1] == 'C' else 1)
        return tanh(tanh_inputs) * sigmoid(sigmoid_inputs)
