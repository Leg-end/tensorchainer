from tensorlib.engine import base_hook
from tensorflow.python import ops
from tensorflow.python.ops import gen_array_ops
from tensorlib.engine import graph_ops
from tensorlib.engine import base_lib as F
from tensorlib.utils.generic_util import to_list
from tensorlib.utils.nest import flatten_list
import numpy as np


def check_input(inputs):
    """
    check tensor in inputs
    [a, [tensor(b), tensor(c)]] -> [tensor(a), [tensor(b), tensor(c)]]
    """
    for i, x in enumerate(inputs):
        if x is None:
            raise ValueError("None values not supported")
        if not F.is_tensor(x):
            if isinstance(x, list):
                check_input(x)
            elif isinstance(x, tuple):
                inputs[i] = list(x)
                check_input(inputs[i])
            elif isinstance(x, (np.ndarray, float, int)):
                inputs[i] = ops.convert_to_tensor(x)
            else:
                raise TypeError("Type of atom unit in inputs must be"
                                " one of [np.ndarray, float, int, Tensor]"
                                ", but received " + str(type(x)))
        # elif not hasattr(x, '_anchor'):
        #     raise ValueError("Missing attribute '_anchor' from %d tensor %s"
        #                      " in inputs, which means missing previous"
        #                      " connection anchor information and connections"
        #                      " in graph-network will be broken. (Note that"
        #                      " tensor should be computed from an instance of Layer)")


class EmptyHook(base_hook.Hook):
    name = 'EmptyHook'


class LocalHook(base_hook.Hook):
    name = 'LocalHook'

    def before_forward(self, layer, inputs, kwargs):
        check_input(inputs)

    def after_forward(self, layer, outputs, inputs, kwargs):
        from tensorlib.engine.base_layer import LayerList
        # Only a graph network or a simple layer can convert to a node
        if hasattr(layer, '_graph') or not isinstance(layer, LayerList):
            graph_ops.build_node(layer, inputs, outputs, kwargs)


class RNNSpecHook(base_hook.Hook):
    name = 'RNNSpecHook'

    def before_forward(self, layer, inputs, kwarg):
        states = inputs[1]
        state_size = to_list(layer.state_size)
        if len(states) != state_size:
            raise ValueError("The states passed to RNN cell %s"
                             " have %d states, but RNN cell %s"
                             " only has %d state_size, the RNN"
                             " cell %s 's state_size should be"
                             " one integer for each RNN state" % (
                              layer.name, len(states), layer.name,
                              len(state_size), layer.name))


class ExtractHook(base_hook.Hook):
    """
        Example:
            Code example::
                hook = ExtractHook(in_names='input', out_names='conv2d2')
                with hook:
                    net = Input(...'input')
                    net = Conv2D(...'conv2d1')(net)
                    net = Conv2D(...'conv2d2')(net)
                    net = Dense(...'dense1')(net)
                inputs, outputs = hook.get_extract()
                network = Network(inputs=inputs, outputs=outputs)
    """
    name = 'ExtractHook'

    def __init__(self, out_names, in_names=None, prefix=''):
        self.in_names = [prefix + name for name in to_list(in_names)] if in_names else []
        if out_names is None:
            raise ValueError("Export layers' names can not be None")
        self.out_names = [prefix + name for name in to_list(out_names)]
        self._inputs = []
        self._outputs = []

    def before_forward(self, layer, inputs, kwargs):
        if self.in_names:
            scope = ops.get_name_scope()
            for tensor in flatten_list(inputs):
                name = scope + '/' + layer.name
                if name in self.in_names:
                    self._inputs.append(tensor)
                    self.in_names.remove(name)

    def after_forward(self, layer, outputs, inputs, kwargs):
        if self.out_names:
            scope = ops.get_name_scope()
            for tensor in flatten_list(outputs):
                name = scope + '/' + layer.name
                if name in self.out_names:
                    self._outputs.append(tensor)
                    self.out_names.remove(name)

    def get_extract(self):
        if len(self.in_names) != 0:
            raise RuntimeError("Can not find input tensor(s) from [%s]"
                               % ','.join(self.in_names))
        if len(self.out_names) != 0:
            raise RuntimeError("Can not find output tensor(s) from:\n%s"
                               % '\n'.join(self.out_names))
        return self._inputs, self._outputs


class NumericHook(base_hook.Hook):
    name = 'NumericHook'

    def after_forward(self, layer, outputs, inputs, kwargs):
        message = ops.get_name_scope() + '/' + layer.name
        for i in range(len(outputs)):
            outputs[i] = gen_array_ops.check_numerics(outputs[i], message=message)
