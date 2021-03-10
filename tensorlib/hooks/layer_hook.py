from tensorlib.engine import base_hook
from tensorflow.python import ops
from tensorlib.engine import graph_ops
from tensorlib.utils.generic_util import to_list
from tensorlib.utils.nest import flatten_list, map_structure
from collections import OrderedDict
import numpy as np


def convert_non_tensor(x):
    if x is None:
        raise ValueError("None values not supported")
    if isinstance(x, (np.ndarray, float, int)):
        return ops.convert_to_tensor(x)
    return x


class EmptyHook(base_hook.Hook):
    name = 'EmptyHook'


class LocalHook(base_hook.Hook):
    name = 'LocalHook'

    def before_forward(self, layer, inputs, **kwargs):
        map_structure(convert_non_tensor, inputs, inplace=True)

    def after_forward(self, layer, outputs, inputs, *args, **kwargs):
        # Only a named network or layer can convert to a node
        # Cause a named network or layer can maintain a name scope
        if layer.name:
            graph_ops.build_node(layer, inputs, outputs, kwargs)


class RNNSpecHook(base_hook.Hook):
    name = 'RNNSpecHook'

    def before_forward(self, layer, inputs, **kwarg):
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
                hook = ExtractHook(end_names=['input', 'conv2d2'])
                with hook:
                    net = Input(...'input')
                    net = Conv2D(...'conv2d1')(net)
                    net = Conv2D(...'conv2d2')(net)
                    net = Dense(...'dense1')(net)
                endpoints = hook.get_endpoints()
                network = Network(inputs=endpoints['input'], outputs=endpoints['conv2d2'])
    """
    name = 'ExtractHook'

    def __init__(self, end_names, prefix=''):
        if not end_names:
            raise ValueError("End points' names must be provided")
        if prefix[-1] != '/':
            prefix += '/'
        self.end_names = [prefix + name for name in to_list(end_names)]
        self.endpoints = OrderedDict()

    def after_forward(self, layer, outputs, inputs, **kwargs):
        scope = ops.get_name_scope()
        for tensor in flatten_list(outputs):
            name = scope + '/' + layer.name if scope else layer.name
            if name in self.end_names:
                self.endpoints[name] = tensor

    def get_endpoints(self):
        missed_names = set(self.end_names) - set(self.endpoints.keys())
        if len(missed_names) > 0:
            raise ValueError("Can not find tensor(s) from:\n%s"
                             % '\n'.join(missed_names))
        return self.endpoints
