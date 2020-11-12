from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from contextlib import contextmanager
from tensorlib.engine.base_layer import Layer, LayerList
from tensorflow.python import ops
from tensorlib.engine import graph_ops
from tensorlib.engine import base_lib as F
from tensorlib.utils.generic_util import to_list, object_list_uid, unpack_singleton
import typing as tp

__all__ = ["Network", "graph_scope"]


class GraphScope:
    def __init__(self, scope, inputs, outputs=None):

        self.scope = scope
        self._inputs = inputs
        self._outputs = outputs
        self._inputs_used = False
        self._outputs_assigned = False

    @property
    def inputs(self):
        self._inputs_used = True
        return self._inputs

    @property
    def outputs(self):
        if not self._outputs_assigned or not self._inputs_used or self._outputs is None:
            raise ValueError("To convert a network inside `graph_scope` to "
                             " a graph network, you must provide this network's"
                             " outputs by setting GraphScope.outputs and use"
                             " GraphScope.inputs as inputs.")
        return self._outputs

    @outputs.setter
    def outputs(self, y):
        self._outputs_assigned = True
        self._outputs = y


@contextmanager
def graph_scope(name, default_name=None, values=None):
    from tensorlib.engine import Input
    if values is None:
        raise ValueError("Argument `values` can not be None.")
    values = to_list(values)
    [F.assert_tensor_traceable(x) for x in values]
    with ops.name_scope(name=name, default_name=default_name,
                        values=values) as scope:
        inputs = unpack_singleton([
            Input(batch_input_shape=F.int_shape(x),
                  dtype=x.dtype)
            for x in values])
        handler = GraphScope(scope=scope, inputs=inputs)
        yield handler
    net = Network(inputs=inputs, outputs=handler.outputs, name=scope)
    graph_ops.build_node(net, values, to_list(handler.outputs))
    # print(getattr(handler.outputs, '_anchor')[0])
    del handler


class Network(Layer):

    @property
    def trainable_weights(self):
        for w in super(Network, self).trainable_weights:
            yield w
        for child in self.children():
            for w in child.trainable_weights:
                yield w

    @property
    def non_trainable_weights(self):
        for w in super(Network, self).non_trainable_weights:
            yield w
        for child in self.children():
            for w in child.non_trainable_weights:
                yield w

    @property
    def weights(self):
        for w in super(Network, self).weights:
            yield w
        for child in self.children():
            for w in child.weights:
                yield w

    def __init__(self, inputs=None, outputs=None, **kwargs):
        self._children = set()
        self._graph = None
        self._output_tensor_cache = {}
        self.inputs = None
        self.outputs = None
        if inputs is not None and outputs is not None:
            super(Network, self).__init__(**kwargs)
            inputs = to_list(inputs)
            outputs = to_list(outputs)
            [F.assert_tensor_traceable(x) for x in inputs + outputs]
            self.inputs = inputs
            self.outputs = outputs
            self._graph = graph_ops.build_graph_network(self, inputs, outputs)
            self.built = True
        else:
            keys = list(kwargs.keys())
            for key in keys:
                if isinstance(kwargs[key], Layer):
                    self.add_layer(key, kwargs[key])
                    kwargs.pop(key)
            super(Network, self).__init__(**kwargs)
            self.local_hooks['LocalHook'].build = False

    def __setattr__(self, name: str, value):
        if isinstance(value, Layer) or (isinstance(value, (list, tuple)) and all(
                isinstance(v, Layer) for v in value)):
            if isinstance(value, (list, tuple)):
                value = LayerList(*value, name='')
            self.check_define_before_run()
            if not hasattr(self, name) or not isinstance(getattr(self, name), Layer):
                self._children.add(name)
            else:
                raise AttributeError("Layer named `%s` already existed" % name)
        super(Network, self).__setattr__(name, value)

    def __delattr__(self, name: str):
        if name in self._children:
            self.check_define_before_run()
            self._children.discard(name)
        super(Network, self).__delattr__(name)

    def layers(self, skip_self: bool = False):
        if not skip_self:
            yield self
        for child in self.children():
            for layer in child.layers():
                yield layer

    def add_layer(self, name: str, layer: Layer):
        setattr(self, name, layer)

    def children(self) -> tp.Iterator[Layer]:
        if self._graph:
            for child in self._graph.children:
                yield child
        else:
            d = self.__dict__
            for name in sorted(self._children):
                yield d[name]

    def get_layer(self, name):
        for layer in self.layers(skip_self=True):
            if name == layer.name:
                return layer
        raise KeyError('Layer named `%s` does not exist' % name)

    def forward(self, *inputs, **kwargs):
        if not self._graph:
            raise NotImplementedError
        else:
            key = object_list_uid(inputs)
            if key in self._output_tensor_cache:
                return self._output_tensor_cache[key]
            else:
                outputs = self._graph(*inputs)
                self._output_tensor_cache[key] = outputs
            return outputs
