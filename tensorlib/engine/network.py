from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from contextlib import contextmanager
from tensorlib.engine.base_layer import Layer, LayerList
from tensorflow.python import ops
from tensorlib.engine import graph_ops
from tensorlib.engine import base_lib as F
from tensorlib.utils.generic_util import to_list, object_list_uid, unpack_singleton
from tensorlib.utils import nest
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
        self._nested_inputs = None
        self._nested_outputs = None
        self.inputs = None
        self.outputs = None
        if inputs is not None and outputs is not None:
            super(Network, self).__init__(**kwargs)
            self.inputs = to_list(nest.flatten(inputs))
            self.outputs = to_list(nest.flatten(outputs))
            [F.assert_tensor_traceable(x)
             for x in self.inputs + self.outputs]
            self._nested_inputs = inputs
            self._nested_outputs = outputs
            self._graph = graph_ops.build_graph_network(self, inputs, outputs)
            self.built = True
        else:
            keys = list(sorted(kwargs.keys()))
            for key in keys:
                if isinstance(kwargs[key], Layer):
                    self.add_layer(key, kwargs.pop(key))
            super(Network, self).__init__(**kwargs)

    def __setattr__(self, name: str, value):
        # Cause computation logic already fixed, we don't check "define before run"
        if isinstance(value, Layer):
            # Ignore Loss and Metric
            if value.__class__.__base__.__name__ in ['Loss', 'Metric']:
                pass
            elif name not in self._children:
                self._children.add(name)
            # Check whether have same direct superclass
            elif not isinstance(value, getattr(self, name).__class__.__base__):
                raise TypeError("Replacing an instance of %s with an instance"
                                " of %s is not allowed." % (
                                    getattr(self, name).__class__.__base__.__name__),
                                value.__class.__base__.__name__)
        elif isinstance(value, (list, tuple)) and len(value) > 0 and all(
                isinstance(v, Layer) for v in value):
            value = LayerList(*value, name='')
            self._children.add(name)
        super(Network, self).__setattr__(name, value)

    def __delattr__(self, name: str):
        if name in self._children:
            self.check_define_before_run()
            self._children.discard(name)
        super(Network, self).__delattr__(name)

    def add_layer(self, name: str, layer: Layer):
        setattr(self, name, layer)

    def get_layer(self, name=None, index=None):
        """
        Retrieves layer based on either its name, or index, but name first
        Retrieve order in topological graph traversal when by index(Note:
        Topological graph must build first)
        :param name: str
        :param index: int
        :return: layer
        """
        if name is not None:
            if name in self._children:
                return getattr(self, name)
            else:
                raise KeyError("No such layer: " + name)
        if self._graph is None:
            raise ValueError("Topological index retrieving only available after"
                             " building a topological graph of this network: " + self.name)
        elif index is None:
            raise ValueError("Provide either name or index for retrieving")
        else:
            assert index >= 0
            if index > len(self._graph.children):
                raise ValueError("Requested to retrieve layer at index: " + str(index)
                                 + ", but model only has " + str(len(self._graph.children)) + ' layers')
            else:
                return self._graph.children[index]

    def children(self) -> tp.Iterator[Layer]:
        if self._graph:
            # foreach in topological order
            for child in sorted(set(self._graph.children), key=self._graph.children.index):
                yield child
        else:
            d = self.__dict__
            for name in sorted(self._children):
                yield d[name]

    def layers(self, skip_self: bool = False):
        if not skip_self:
            yield self
        for child in self.children():
            for layer in child.layers():
                yield layer

    def forward(self, *inputs, **kwargs):
        if not self._graph:
            raise NotImplementedError
        else:
            key = object_list_uid(inputs)
            if key in self._output_tensor_cache:
                return self._output_tensor_cache[key]
            else:
                outputs = self._graph(inputs)
                self._output_tensor_cache[key] = outputs
            return outputs
