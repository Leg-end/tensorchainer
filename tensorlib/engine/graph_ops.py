from tensorlib.utils.generic_util import to_list, unpack_singleton, has_arg
from tensorlib.utils import nest
from collections import namedtuple
import tensorlib

from tensorflow.python import ops


class History(namedtuple('History', ['node', 'tensor_index'])):
    __slots__ = ()


class Node(object):

    def __init__(self,
                 layer,
                 in_degree,
                 tensor_indices,
                 input_tensors,
                 output_tensors,
                 arguments=None):
        self.layer = layer
        layer.add_mirror(self)
        self.name = layer.name + ':' + str(len(layer.mirrors))
        self.in_degree = in_degree
        self.out_degree = []
        self.tensor_indices = tensor_indices  # nested (e.g. [[1, 2], 3])
        self.input_tensors = input_tensors  # nested (e.g. [[1, 2], 3])
        self.output_tensors = output_tensors  # nested (e.g. [[1, 2], 3])
        self.arguments = arguments or {}
        for node in self.in_degree:
            if node:
                node.out_degree.append(self)

    def __repr__(self):
        return self.layer.name

    def __call__(self, inputs):
        hooks = tensorlib._get_hooks()
        hooks = hooks.values()
        for hook in hooks:
            hook.before_forward(self.layer, inputs, **self.arguments)
        with ops.name_scope(self.layer.name):
            outputs = self.layer.forward(*inputs, **self.arguments)
            # outputs return from __call__ should have same format
            # as from forward, e.g. outputs from forward is [?]
            unpack = not isinstance(outputs, (list, tuple))
            outputs = to_list(outputs)
        for hook in hooks:
            hook.after_forward(self.layer, outputs, inputs, **self.arguments)
        if unpack:
            outputs = outputs[0]
        return outputs


class GraphNetwork(object):

    @property
    def depth(self):
        return len(self.nodes)

    @property
    def count(self):
        num = 0
        for d_nodes in self.nodes:
            num += len(d_nodes)
        return num

    def __init__(self,
                 network,
                 nodes,
                 children):
        self.network = network
        self.nodes = nodes
        self.children = children  # topological order layers

    def __repr__(self):
        return str(self.nodes)

    def __call__(self, inputs):
        # All feed operations do in flatten
        inputs = nest.flatten(inputs)
        log = {str(id(x)): y for x, y in zip(nest.flatten(
            self.network._nested_inputs), inputs)}
        # Ignore the InputLayers when computing the graph.
        for d_nodes in self.nodes[1:]:
            for node in d_nodes:
                feed_tensors = nest.map_structure(
                    lambda t: log[str(id(t))], node.input_tensors)
                output_tensors = node(feed_tensors)
                for x, y in zip(nest.flatten(
                        node.output_tensors), nest.flatten(output_tensors)):
                    log[str(id(x))] = y
        # Export outputs according to its structure
        outputs = nest.map_structure(lambda t: log[str(id(t))],
                                     self.network._nested_inputs)
        return unpack_singleton(outputs)


def build_graph_network(network, inputs, outputs):
    inputs = to_list(inputs)
    outputs = to_list(outputs)
    children = list()  # topological order
    nodes = set()
    # In order to maintain depth information of nodes,
    # we decide not to use topological structure to store nodes,
    # instead, we use a nested list to store nodes depth-wise
    nodes_by_depth = []
    out_nodes = [getattr(x, '_history').node for x in nest.flatten_list(outputs)]
    for node in out_nodes:  # remove useless nodes in out_node's out_degree
        node.out_degree = list(set(node.out_degree) & set(out_nodes))
    cur_depth = []
    # Assume we already visited all current depth nodes' in-degree nodes
    for tensor in nest.flatten_list(inputs):
        node = getattr(tensor, '_history').node
        cur_depth.append(node)
        nodes |= set(node.in_degree)
    while cur_depth:
        next_depth = []
        nodes_by_depth.append([])
        for node in cur_depth:
            if node in nodes or any(n not in nodes for n in node.in_degree):
                continue
            nodes_by_depth[-1].append(node)
            nodes.add(node)
            children.append(node.layer)
            next_depth.extend(node.out_degree)
        cur_depth = next_depth
    return GraphNetwork(network=network,
                        nodes=nodes_by_depth,
                        children=children)


def build_node(layer, inputs: (list, tuple), outputs: (list, tuple), arguments=None):
    in_degree = [getattr(tensor, '_history', History(None, 0)).node
                 for tensor in nest.flatten_list(inputs)]
    uses_lp = any(getattr(tensor, '_uses_learning_phase', False)
                  for tensor in nest.flatten_list(inputs)) or has_arg(layer.forward, 'training')
    tensor_indices = nest.map_structure(lambda x: getattr(
        x, '_history', History(None, 0)).tensor_index, inputs)
    node = Node(layer=layer,
                in_degree=in_degree,
                tensor_indices=tensor_indices,
                input_tensors=inputs,
                output_tensors=outputs,
                arguments=arguments)
    for i, tensor in enumerate(nest.flatten_list(outputs)):
        setattr(tensor, '_history', History(node, i))
        setattr(tensor, '_uses_learning_phase', uses_lp)
