from tensorlib.engine.base_layer import Layer, param_tracker
from tensorlib.utils.conv_util import *
from tensorlib.utils.generic_util import to_list
from tensorlib.engine import base_lib as F
from tensorlib.engine.scope_manager import add_arg_scope

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import nn


__all__ = ['MaxPool1D', 'AvgPool1D', 'AdaptiveMaxPool1D', 'AdaptiveAvgPool1D',
           'MaxPool2D', 'AvgPool2D', 'AdaptiveMaxPool2D', 'AdaptiveAvgPool2D',
           'MaxPool3D', 'AvgPool3D', 'AdaptiveMaxPool3D', 'AdaptiveAvgPool3D',
           'GlobalMaxPool', 'GlobalAvgPool', 'ROIPooling']


class GlobalMaxPool(Layer):
    @param_tracker()
    @add_arg_scope
    def __init__(self,
                 data_format='channels_last',
                 keepdims=True,
                 **kwargs):
        super(GlobalMaxPool, self).__init__(**kwargs)
        self.data_format = normalize_data_format(data_format, 2)
        self.keepdims = keepdims
        self.axes = None

    def build(self, input_shape):
        self.axes = list(range(1, len(input_shape)))
        if self.data_format[-1] == 'C':
            self.axes.pop(-1)
        else:
            self.axes.pop(0)

    def forward(self, inputs):
        return math_ops.reduce_max(inputs, axis=self.axes,
                                   keepdims=self.keepdims, name='pool_block')


class GlobalAvgPool(Layer):
    @param_tracker()
    @add_arg_scope
    def __init__(self,
                 data_format='channels_last',
                 keepdims=True,
                 **kwargs):
        super(GlobalAvgPool, self).__init__(**kwargs)
        self.data_format = normalize_data_format(data_format, 2)
        self.keepdims = keepdims
        self.axes = None

    def build(self, input_shape):
        self.axes = list(range(1, len(input_shape)))
        if self.data_format[-1] == 'C':
            self.axes.pop(-1)
        else:
            self.axes.pop(0)

    def forward(self, inputs):
        return math_ops.reduce_mean(inputs, axis=self.axes,
                                    keepdims=self.keepdims, name='pool_block')


class Pool(Layer):
    def __init__(self,
                 rank,
                 kernel_size,
                 pool_function,
                 strides=2,
                 padding='VALID',
                 data_format='channels_last',
                 **kwargs):
        super(Pool, self).__init__(**kwargs)
        assert isinstance(rank, int) and 3 >= rank >= 1
        assert callable(pool_function)
        self.rank = rank
        self.pool_function = pool_function
        self.data_format = normalize_data_format(data_format, rank)
        self.kernel_size = normalize_tuple(kernel_size, rank, 'kernel_size')
        self.strides = normalize_tuple(strides, rank, 'strides')
        self.padding = normalize_padding(padding)

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class AdaptivePool(Layer):
    def __init__(self,
                 rank,
                 out_size,
                 pool_function,
                 data_format='channels_last',
                 **kwargs):
        super(AdaptivePool, self).__init__(**kwargs)
        assert isinstance(rank, int) and 3 >= rank >= 1
        self.rank = rank
        self.pool_function = pool_function
        self.out_size = normalize_tuple(out_size, rank, 'out_size')
        self.data_format = normalize_data_format(data_format, rank)
        self.kernel_size = None
        self.strides = (1,) * rank
        self.padding = 'VALID'

    def build(self, input_shape):
        if len(input_shape) > self.rank + 2:
            raise ValueError("Input to {} should has {:d} rank,"
                             "but received input shape {}".format(
                              self.name, self.rank, str(input_shape)))
        spatial = to_list(input_shape[1:-1]) \
            if self.data_format[-1] == 'C' else to_list(input_shape[2:])
        self.kernel_size = tuple(spatial[i] - (self.out_size[i] - 1) * self.strides[0]
                                 for i in range(self.rank))

    def forward(self, inputs):
        raise NotImplementedError


class Pool1D(Pool):

    def __init__(self,
                 kernel_size,
                 pool_function,
                 strides=(2,),
                 padding='VALID',
                 data_format='channels_last',
                 **kwargs):
        super(Pool1D, self).__init__(rank=1,
                                     kernel_size=kernel_size,
                                     pool_function=pool_function,
                                     strides=strides,
                                     padding=padding,
                                     data_format=data_format,
                                     **kwargs)

    def forward(self, inputs):
        if self.data_format[-1] == 'C':
            pad_axis = 2
            kernel_size = (1,) + self.kernel_size + (1, 1)
            strides = (1,) + self.strides + (1, 1)
            data_format = 'NHWC'
        else:
            pad_axis = 3
            kernel_size = (1, 1) + self.kernel_size + (1,)
            strides = (1, 1) + self.strides + (1,)
            data_format = 'NCHW'
        inputs = array_ops.expand_dims(inputs, pad_axis)
        outputs = self.pool_function(
            inputs,
            kernel_size,
            strides=strides,
            padding=self.padding,
            data_format=data_format)
        return array_ops.squeeze(outputs, pad_axis)


class AdaptivePool1D(AdaptivePool):

    def __init__(self,
                 out_size,
                 pool_function,
                 data_format='channels_last',
                 **kwargs):
        super(AdaptivePool1D, self).__init__(rank=1,
                                             out_size=out_size,
                                             pool_function=pool_function,
                                             data_format=data_format,
                                             **kwargs)

    def forward(self, inputs):
        if self.data_format[-1] == 'C':
            pad_axis = 2
            kernel_size = (1,) + self.kernel_size + (1, 1)
            strides = (1,) + self.strides + (1, 1)
        else:
            pad_axis = 3
            kernel_size = (1, 1) + self.kernel_size + (1,)
            strides = (1, 1) + self.strides + (1,)
        inputs = array_ops.expand_dims(inputs, pad_axis)
        outputs = self.pool_function(
            inputs,
            kernel_size,
            strides=strides,
            padding=self.padding,
            data_format=self.data_format)
        return array_ops.squeeze(outputs, pad_axis)


class MaxPool1D(Pool1D):
    @param_tracker()
    @add_arg_scope
    def __init__(self,
                 kernel_size,
                 strides=(2,),
                 padding='VALID',
                 data_format='channels_last',
                 **kwargs):
        super(MaxPool1D, self).__init__(kernel_size=kernel_size,
                                        pool_function=nn.max_pool,
                                        strides=strides,
                                        padding=padding,
                                        data_format=data_format,
                                        **kwargs)


class AvgPool1D(Pool1D):
    @param_tracker()
    @add_arg_scope
    def __init__(self,
                 kernel_size,
                 strides=(2,),
                 padding='VALID',
                 data_format='channels_last',
                 **kwargs):
        super(AvgPool1D, self).__init__(kernel_size=kernel_size,
                                        pool_function=nn.avg_pool,
                                        strides=strides,
                                        padding=padding,
                                        data_format=data_format,
                                        **kwargs)


class AdaptiveMaxPool1D(AdaptivePool1D):
    @param_tracker()
    @add_arg_scope
    def __init__(self,
                 out_size,
                 data_format='channels_last',
                 **kwargs):
        super(AdaptiveMaxPool1D, self).__init__(out_size=out_size,
                                                pool_function=nn.max_pool,
                                                data_format=data_format,
                                                **kwargs)


class AdaptiveAvgPool1D(AdaptivePool1D):
    @param_tracker()
    @add_arg_scope
    def __init__(self,
                 out_size,
                 data_format='channels_last',
                 **kwargs):
        super(AdaptiveAvgPool1D, self).__init__(out_size=out_size,
                                                pool_function=nn.avg_pool,
                                                data_format=data_format,
                                                **kwargs)


class Pool2D(Pool):

    def __init__(self,
                 kernel_size,
                 pool_function,
                 strides=(2, 2),
                 padding='VALID',
                 data_format='channels_last',
                 **kwargs):
        super(Pool2D, self).__init__(rank=2,
                                     kernel_size=kernel_size,
                                     pool_function=pool_function,
                                     strides=strides,
                                     padding=padding,
                                     data_format=data_format,
                                     **kwargs)

    def forward(self, inputs):
        if self.data_format[-1] == 'C':
            kernel_size = (1,) + self.kernel_size + (1,)
            strides = (1,) + self.strides + (1,)
        else:
            kernel_size = (1, 1) + self.kernel_size
            strides = (1, 1) + self.strides
        outputs = self.pool_function(
            inputs,
            ksize=kernel_size,
            strides=strides,
            padding=self.padding,
            data_format=self.data_format)
        return outputs


class AdaptivePool2D(AdaptivePool):

    def __init__(self,
                 out_size,
                 pool_function,
                 data_format='channels_last',
                 **kwargs):
        super(AdaptivePool2D, self).__init__(rank=2,
                                             out_size=out_size,
                                             pool_function=pool_function,
                                             data_format=data_format,
                                             **kwargs)

    def forward(self, inputs):
        if self.data_format[-1] == 'C':
            kernel_size = (1,) + self.kernel_size + (1,)
            strides = (1,) + self.strides + (1,)
        else:
            kernel_size = (1, 1) + self.kernel_size
            strides = (1, 1) + self.strides
        outputs = self.pool_function(
            inputs,
            ksize=kernel_size,
            strides=strides,
            padding=self.padding,
            data_format=self.data_format)
        return outputs


class MaxPool2D(Pool2D):
    @param_tracker()
    @add_arg_scope
    def __init__(self,
                 kernel_size,
                 strides=(2, 2),
                 padding='VALID',
                 data_format='channels_last',
                 **kwargs):
        super(MaxPool2D, self).__init__(kernel_size=kernel_size,
                                        pool_function=nn.max_pool,
                                        strides=strides,
                                        padding=padding,
                                        data_format=data_format,
                                        **kwargs)


class AvgPool2D(Pool2D):
    @param_tracker()
    @add_arg_scope
    def __init__(self,
                 kernel_size,
                 strides=(2, 2),
                 padding='VALID',
                 data_format='channels_last',
                 **kwargs):
        super(AvgPool2D, self).__init__(kernel_size=kernel_size,
                                        pool_function=nn.avg_pool,
                                        strides=strides,
                                        padding=padding,
                                        data_format=data_format,
                                        **kwargs)


class AdaptiveMaxPool2D(AdaptivePool1D):
    @param_tracker()
    @add_arg_scope
    def __init__(self,
                 out_size,
                 data_format='channels_last',
                 **kwargs):
        super(AdaptiveMaxPool2D, self).__init__(out_size=out_size,
                                                pool_function=nn.max_pool,
                                                data_format=data_format,
                                                **kwargs)


class AdaptiveAvgPool2D(AdaptivePool1D):
    @param_tracker()
    @add_arg_scope
    def __init__(self,
                 out_size,
                 data_format='channels_last',
                 **kwargs):
        super(AdaptiveAvgPool2D, self).__init__(out_size=out_size,
                                                pool_function=nn.avg_pool,
                                                data_format=data_format,
                                                **kwargs)


class Pool3D(Pool):

    def __init__(self,
                 kernel_size,
                 pool_function,
                 strides=(2, 2, 2),
                 padding='VALID',
                 data_format='channels_last',
                 **kwargs):
        super(Pool3D, self).__init__(rank=3,
                                     kernel_size=kernel_size,
                                     pool_function=pool_function,
                                     strides=strides,
                                     padding=padding,
                                     data_format=data_format,
                                     **kwargs)

    def forward(self, inputs):
        kernel_size = (1,) + self.kernel_size + (1,)
        strides = (1,) + self.strides + (1,)
        if self.data_format[-1] != 'C':
            inputs = F.transpose_to_channels_last(inputs)
        outputs = self.pool_function(
            inputs,
            ksize=kernel_size,
            strides=strides,
            padding=self.padding)
        if self.data_format[-1] != 'C':
            outputs = F.transpose_to_channels_first(outputs)
        return outputs


class AdaptivePool3D(AdaptivePool):

    def __init__(self,
                 out_size,
                 pool_function,
                 data_format='channels_last',
                 **kwargs):
        super(AdaptivePool3D, self).__init__(rank=3,
                                             out_size=out_size,
                                             pool_function=pool_function,
                                             data_format=data_format,
                                             **kwargs)

    def forward(self, inputs):
        kernel_size = (1,) + self.kernel_size + (1,)
        strides = (1,) + self.strides + (1,)
        if self.data_format[-1] != 'C':
            inputs = F.transpose_to_channels_last(inputs)
        outputs = self.pool_function(
            inputs,
            ksize=kernel_size,
            strides=strides,
            padding=self.padding)
        if self.data_format[-1] != 'C':
            outputs = F.transpose_to_channels_first(outputs)
        return outputs


class MaxPool3D(Pool3D):
    @param_tracker()
    @add_arg_scope
    def __init__(self,
                 kernel_size,
                 strides=(2, 2, 2),
                 padding='VALID',
                 data_format='channels_last',
                 **kwargs):
        super(MaxPool3D, self).__init__(kernel_size=kernel_size,
                                        pool_function=nn.max_pool3d,
                                        strides=strides,
                                        padding=padding,
                                        data_format=data_format,
                                        **kwargs)


class AvgPool3D(Pool3D):
    @param_tracker()
    @add_arg_scope
    def __init__(self,
                 kernel_size,
                 strides=(2, 2, 2),
                 padding='VALID',
                 data_format='channels_last',
                 **kwargs):
        super(AvgPool3D, self).__init__(kernel_size=kernel_size,
                                        pool_function=nn.avg_pool3d,
                                        strides=strides,
                                        padding=padding,
                                        data_format=data_format,
                                        **kwargs)


class AdaptiveMaxPool3D(AdaptivePool1D):
    @param_tracker()
    @add_arg_scope
    def __init__(self,
                 out_size,
                 data_format='channels_last',
                 **kwargs):
        super(AdaptiveMaxPool3D, self).__init__(out_size=out_size,
                                                pool_function=nn.max_pool3d,
                                                data_format=data_format,
                                                **kwargs)


class AdaptiveAvgPool3D(AdaptivePool1D):
    @param_tracker()
    @add_arg_scope
    def __init__(self,
                 out_size,
                 data_format='channels_last',
                 **kwargs):
        super(AdaptiveAvgPool3D, self).__init__(out_size=out_size,
                                                pool_function=nn.avg_pool3d,
                                                data_format=data_format,
                                                **kwargs)


class ROIPooling(Layer):
    @param_tracker()
    @add_arg_scope
    def __init__(self,
                 crop_size,
                 data_format='channels_last',
                 **kwargs):
        super(ROIPooling, self).__init__(**kwargs)
        self.crop_size = normalize_tuple(crop_size, 2, 'crop_size')
        self.data_format = normalize_data_format(data_format, 2)
        self._kernel_size = (2, 2)
        self._strides = (2, 2)

    def forward(self, inputs, rois, im_info):
        """
        :param inputs: features
        :param rois: regions of interest, shape with [batch, 5]
            format as (img_id, x0, y0, x1, y1)
            img_id is index of image inside batch
        :param im_info: scales of image, shape with [batch, 2]
            format as (height, width)
        :return:
        """
        assert F.ndim(rois) == 2 and F.int_shape(rois)[-1] == 5
        assert F.ndim(im_info) == 2 and F.int_shape(im_info)[-1] == 2
        indices = F.int32(rois[:, 0])
        boxes = rois[:, 1:]
        norm = F.float32(array_ops.stack([
            im_info[:, 1], im_info[:, 0],
            im_info[:, 1], im_info[:, 0]], axis=1))
        boxes = boxes / norm
        # (x0, y0, x1, y1) -> (y0, x0, y1, x1)
        boxes = array_ops.stack([
            boxes[:, 1], boxes[:, 0],
            boxes[:, 3], boxes[:, 2]], axis=1)
        crop_size = array_ops.constant(self.crop_size)
        if self.data_format[-1] == 'C':
            kernel_size = (1,) + self._kernel_size + (1,)
            strides = (1,) + self._strides + (1,)
        else:
            kernel_size = (1, 1) + self._kernel_size
            strides = (1, 1) + self._strides
            inputs = F.transpose_to_channels_last(inputs)
        outputs = image_ops.crop_and_resize(
            image=inputs, boxes=boxes,
            box_ind=indices, crop_size=crop_size)
        if self.data_format[-1] != 'C':
            outputs = F.transpose_to_channels_first(outputs)
        outputs = nn.max_pool2d(
            input=outputs,
            ksize=kernel_size,
            strides=strides,
            data_format=self.data_format,
            padding='SAME')
        return outputs
