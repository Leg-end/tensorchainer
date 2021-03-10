import tensorlib as lib
import tensorflow as tf
import functools
import collections


def _split_divisible(num, num_ways, divisible_by=8):
    assert num % divisible_by == 0
    assert num / num_ways >= divisible_by
    # Note: want to round down, we adjust each split to match the total.
    base = num // num_ways // divisible_by * divisible_by
    result = []
    accumulated = 0
    for i in range(num_ways):
        r = base
        while accumulated + r < num * (i + 1) / num_ways:
            r += divisible_by
        result.append(r)
        accumulated += r
    assert accumulated == num
    return result


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class Block(collections.namedtuple('Block', ['scope', 'object', 'args'])):
    """
    """


def expand_input_by_factor(n, divisible_by=8):
    return lambda num_inputs, **_: _make_divisible(num_inputs * n, divisible_by)


def depth_multiplier(multiplier,
                     d,
                     divisible_by=8,
                     min_depth=8):
    out_channels = _make_divisible(d * multiplier, divisible_by, min_depth)
    return out_channels


def mobile_v2_conv(scope, kernel_size, out_channels, stride):
    multiplier_func = functools.partial(
        depth_multiplier, d=out_channels, divisible_by=8, min_depth=8)

    return Block(scope, Conv2D, [{
        'stride': stride,
        'kernel_size': kernel_size,
        'out_channels': out_channels,
        'store_endpoint': True,
        'multiplier_func': multiplier_func}])


def mobile_v2_blocks(scope, stride, num_units,
                     out_channels, split_expansion=1, expansion_factor=None):
    multiplier_func = functools.partial(
         depth_multiplier, d=out_channels, divisible_by=8, min_depth=8)

    if expansion_factor is None:
        expansion_factor = expand_input_by_factor(6)

    return Block(scope, BottleNeck, [
        {'expansion_factor': expansion_factor,
         'split_expansion': split_expansion,
         'out_channels': out_channels,
         'stride': stride,
         'multiplier_func': multiplier_func,
         'shortcut': True}] +
                 [{'expansion_factor': expansion_factor,
                   'split_expansion': split_expansion,
                   'out_channels': out_channels,
                   'stride': 1,
                   'multiplier_func': multiplier_func,
                   'shortcut': True}] * (num_units - 1))


class FixPadding(lib.engine.Layer):

    def __init__(self,
                 kernel_size,
                 rate=1,
                 name=None):
        super(FixPadding, self).__init__(name=name)
        self.kernel_size = kernel_size
        self.rate = rate
        self.built = True

    def build(self, *args, **kwargs):
        pass

    def forward(self, inputs):
        kernel_size_effective = [
            self.kernel_size[0] + (self.kernel_size[0] - 1) * (self.rate - 1),
            self.kernel_size[0] + (self.kernel_size[0] - 1) * (self.rate - 1)]
        pad_total = [kernel_size_effective[0] - 1, kernel_size_effective[1] - 1]

        pad_beg = [pad_total[0] // 2, pad_total[1] // 2]
        pad_end = [pad_total[0] - pad_beg[0], pad_total[1] - pad_beg[1]]
        padded_inputs = tf.pad(inputs,
                               [[0, 0], [pad_beg[0], pad_end[0]],
                                [pad_beg[1], pad_end[1]], [0, 0]])
        return padded_inputs


class DepthWiseConv(lib.engine.Network):
    def __init__(self,
                 kernel_size=(1, 1),
                 depthwise_multiplier=1,
                 stride=1,
                 padding="SAME",
                 rate=1,
                 activation=None,
                 normalizer_fn=None,
                 normalizer_params=None,
                 name="depthwise_convolution"):
        super(DepthWiseConv, self).__init__(name=name)

        self.activation = activation
        self.depth_wise_conv = lib.layers.DepthWiseConv2D(
            kernel_size=kernel_size, dilation_rate=(rate, rate),
            strides=stride, activation=None, depth_multiplier=depthwise_multiplier,
            use_bias=False,
            bias_initializer=None, padding=padding)
        self.normalizer_fn = None
        if normalizer_fn is not None:
            if not issubclass(normalizer_fn, lib.engine.Layer):
                raise TypeError('Unexpected type of `normalizer_fn`!')
            normalizer_params = normalizer_params or {}
            self.normalizer_fn = normalizer_fn(name=self.depth_wise_conv.name + '/batch_norm',
                                               **normalizer_params)

    def forward(self, inputs):
        if self.normalizer_fn is not None:
            self.normalizer_fn.activation = self.activation
            outputs = self.normalizer_fn(self.depth_wise_conv(inputs))
        else:
            self.depth_wise_conv.activation = self.activation
            outputs = self.depth_wise_conv(inputs)
        return outputs


class Conv2D(lib.engine.Network):
    def __init__(self,
                 out_channels,
                 in_channels,
                 kernel_size=(1, 1),
                 stride=1,
                 padding="SAME",
                 rate=1,
                 activation=tf.nn.relu6,
                 normalizer_fn=lib.layers.BatchNorm,
                 normalizer_params=None,
                 store_endpoint=False,
                 name="convolution"):
        super(Conv2D, self).__init__(name=name)
        self.store_endpoint = store_endpoint
        self._endpoints = {}
        self.activation = activation
        self.conv2d = lib.layers.Conv2D(
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation_rate=(rate, rate),
            strides=stride,
            activation=None,
            use_bias=False,
            padding=padding)

        self.normalizer_fn = None
        if normalizer_fn is not None:
            if not issubclass(normalizer_fn, lib.engine.Layer):
                raise TypeError('Unexpected type of `normalizer_fn`!')
            normalizer_params = normalizer_params or {}
            self.normalizer_fn = normalizer_fn(name=self.conv2d.name + '/batch_norm',
                                               **normalizer_params)

    @property
    def endpoints(self):
        return self._endpoints

    def forward(self, inputs):
        if self.normalizer_fn is not None:
            self.normalizer_fn.activation = self.activation
            outputs = self.normalizer_fn(self.conv2d(inputs))
        else:
            self.conv2d.activation = self.activation
            outputs = self.conv2d(inputs)
        if self.store_endpoint:
            self._endpoints["output"] = outputs
        return outputs


class SplitConv(lib.engine.Network):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_ways=1,
                 stride=(1, 1),
                 divisible_by=8,
                 normalizer_fn=None,
                 activation=None,
                 name="split_conv"):
        super(SplitConv, self).__init__(name=name)
        if in_channels == 1 or min(in_channels // num_ways, out_channels // num_ways) < divisible_by:
            self.conv_layer = Conv2D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=stride,
                normalizer_fn=normalizer_fn,
                activation=activation)
            self.is_split = False
        else:
            self.split_layer = lib.layers.Lambda(
                function=tf.split,
                arguments={'num_or_size_splits': _split_divisible(
                    in_channels, num_ways, divisible_by=divisible_by),
                           'axis': 3},
                name="split")
            self.conv_layers = lib.LayerList()
            for i, out_size in enumerate(_split_divisible(
                    out_channels, num_ways, divisible_by=divisible_by)):
                self.conv_layers.append(Conv2D(
                    in_channels=in_channels,
                    out_channels=out_size,
                    kernel_size=(1, 1),
                    stride=stride,
                    normalizer_fn=normalizer_fn,
                    activation=activation,
                    name="part_%d" % i))
            self.concat_layer = lib.layers.Concat(axis=-1, activation=None)
            self.is_split = True

    def forward(self, inputs):
        conv_outputs = []
        if self.is_split:
            conv_inputs = self.split_layer(inputs)
            for (tensor, layer) in zip(conv_inputs, self.conv_layers):
                conv_outputs.append(layer(tensor))
            return self.concat_layer(*conv_outputs)
        else:
            return self.conv_layer(inputs)


class BottleNeck(lib.engine.Network):
    def __init__(self,
                 out_channels,
                 in_channels,
                 kernel_size=(3, 3),
                 stride=1,
                 rate=1,
                 depthwise_multiplier=1,
                 padding="SAME",
                 shortcut=True,
                 depthwise_location="expansion",
                 explicit_padding=False,
                 expansion_factor=expand_input_by_factor(6),
                 split_projection=1,
                 split_expansion=1,
                 split_divisible_by=8,
                 **kwargs):
        super(BottleNeck, self).__init__(**kwargs)
        self.padding_layer = None
        self.expansion_layer = None
        self._endpoints = {}
        self.in_channels = in_channels
        self.shortcut = shortcut
        self.stride = stride

        if explicit_padding:
            self.padding_layer = FixPadding(kernel_size, rate=rate)
        if depthwise_location not in ['input', 'expansion', 'output']:
            raise TypeError('%s is unknown value of `depthwise_location`' % depthwise_location)
        self.depthwise_location = depthwise_location

        if explicit_padding:
            if padding != 'SAME':
                raise TypeError('`explicit_padding` should only be used with '
                                '`SAME` padding.')

        self.depthwise_layer = DepthWiseConv(
            kernel_size=kernel_size, rate=rate, stride=stride,
            depthwise_multiplier=depthwise_multiplier, padding=padding,
            activation=None, normalizer_fn=lib.layers.BatchNorm)

        if self.depthwise_location == "input":
            self.in_channels *= depthwise_multiplier

        if callable(expansion_factor):
            inner_size = expansion_factor(num_inputs=in_channels)
        else:
            inner_size = expansion_factor
        if inner_size > self.in_channels:
            self.expansion_layer = SplitConv(
                out_channels=inner_size,
                in_channels=self.in_channels,
                num_ways=split_expansion,
                divisible_by=split_divisible_by,
                normalizer_fn=lib.layers.BatchNorm,
                activation=tf.nn.relu6,
                name="split_expansion")
            self.in_channels = inner_size

        if self.depthwise_location == "expansion":
            self.depthwise_layer.activation = tf.nn.relu6
            self.in_channels *= depthwise_multiplier

        self.projection_layer = SplitConv(
            out_channels=out_channels,
            in_channels=self.in_channels,
            num_ways=split_projection,
            stride=1,
            normalizer_fn=lib.layers.BatchNorm,
            activation=None,
            name="split_projection")
        self.add_layer = lib.layers.ElementWise(element_op='add', name='shortcut_output')

    @property
    def endpoints(self):
        return self._endpoints

    def forward(self, inputs):
        net = inputs
        if self.depthwise_location == "input":
            if self.padding_layer:
                net = self.padding_layer(net)
            net = self.depthwise_layer(net)

        if self.expansion_layer:
            net = self.expansion_layer(net)
            self._endpoints["expansion_output"] = net

        if self.depthwise_location == "expansion":
            if self.padding_layer:
                net = self.padding_layer(net)
            net = self.depthwise_layer(net)
            self._endpoints["depthwise_output"] = net

        net = self.projection_layer(net)
        self._endpoints["projection_output"] = net

        if self.depthwise_location == "output":
            if self.padding_layer:
                net = self.padding_layer(net)
            net = self.depthwise_layer(net)

        if self.shortcut and self.stride == 1 and\
                lib.engine.int_shape(net)[3] == lib.engine.int_shape(inputs)[3]:
            net = self.add_layer(net, inputs)
        self._endpoints['output'] = net
        return net
