from tensorlib.engine.base_layer import Layer, param_tracker
from tensorlib.utils.conv_util import normalize_data_format
from tensorlib.engine.scope_manager import add_arg_scope
from tensorlib.utils.generic_util import to_tuple
from tensorflow.python.ops import array_ops


__all__ = ['Flatten', 'flatten',
           'Reshape', 'reshape',
           'Transpose', 'transpose',
           'Tile', 'tile']


class Tile(Layer):
    @param_tracker()
    @add_arg_scope
    def __init__(self,
                 multiples,
                 activation=None,
                 **kwargs):
        super(Tile, self).__init__(activation=activation, **kwargs)
        assert isinstance(multiples, (list, tuple))
        self.multiples = multiples

    def build(self, input_shape):
        if len(self.multiples) != len(input_shape):
            raise ValueError("Expect arg 'multiples' has length {:d},"
                             " but received {}".format(
                              len(input_shape), len(self.multiples)))

    def forward(self, inputs):
        outputs = array_ops.tile(inputs, self.multiples)
        if self.activation:
            outputs = self.activation(outputs)
        return outputs


def tile(inputs, multiples, **kwargs):
    return Tile(multiples=multiples,
                **kwargs)(inputs)


class Flatten(Layer):
    @param_tracker()
    @add_arg_scope
    def __init__(self,
                 data_format='channels_last',
                 activation=None,
                 **kwargs):
        super(Flatten, self).__init__(activation=activation, **kwargs)
        self.data_format = normalize_data_format(data_format, 2)
        self.shape = None
        self.perm = None

    def build(self, input_shape):
        none_count = input_shape.count(None)
        if none_count >= 2:
            raise ValueError("Can not compute flatten shape when missing two"
                             " value in input shape, input shape {}".format(str(input_shape)))
        if self.data_format[-1] != 'C' and len(
                input_shape) is not None and len(input_shape) > 1:
            self.perm = [0]
            self.perm += list(range(2, len(input_shape)))
            self.perm.append(1)
        if none_count == 0:
            self.shape = (input_shape[0], -1)
        else:
            idx = input_shape.index(None)
            if idx > 0:
                self.shape = (input_shape[0], -1)
            else:
                tmp = 1
                for v in input_shape[1:]:
                    tmp *= v
                self.shape = (-1, tmp)

    def forward(self, inputs):
        if self.perm:
            inputs = array_ops.transpose(inputs, perm=self.perm)
        outputs = array_ops.reshape(inputs, self.shape)
        if self.activation:
            outputs = self.activation(outputs)
        return outputs


def flatten(inputs, data_format='channels_last', **kwargs):
    return Flatten(data_format=data_format,
                   **kwargs)(inputs)


class Reshape(Layer):
    @param_tracker()
    @add_arg_scope
    def __init__(self,
                 shape,
                 activation=None,
                 **kwargs):
        super(Reshape, self).__init__(activation=activation, **kwargs)
        if shape is None:
            raise ValueError("Shape for reshaping must be specified.")
        self.shape = to_tuple(shape)

    def forward(self, inputs, shape=None):
        outputs = array_ops.reshape(inputs, self.shape)
        if self.activation:
            outputs = self.activation(outputs)
        return outputs


def reshape(inputs, shape, **kwargs):
    return Reshape(shape=shape,
                   **kwargs)(inputs)


class Transpose(Layer):
    @param_tracker()
    @add_arg_scope
    def __init__(self,
                 perm,
                 activation=None,
                 **kwargs):
        super(Transpose, self).__init__(activation=activation, **kwargs)
        if perm is None:
            raise ValueError("Permutation for Transpose must be specified.")
        self.perm = to_tuple(perm)

    def build(self, input_shape):
        if len(input_shape) != len(self.perm):
            raise ValueError("Found len of 'perm' is {:d} which is"
                             " incompatible with len of 'input_shape': {:d}".format(
                              len(self.perm), len(input_shape)))
        self._built = True

    def forward(self, inputs):
        outputs = array_ops.transpose(inputs, self.perm)
        if self.activation:
            outputs = self.activation(outputs)
        return outputs


def transpose(inputs, perm, **kwargs):
    return Transpose(perm,
                     **kwargs)(inputs)
