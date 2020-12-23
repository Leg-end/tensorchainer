from tensorlib.utils.generic_util import to_list, unpack_singleton, has_arg
from tensorlib.utils.nest import nest_indices, flatten_list
from tensorflow.python.util import nest
import tensorlib

from tensorflow.python import ops


class Node(object):

    def __init__(self,
                 layer,
                 in_degree,
                 tensor_ids,
                 arguments=None):
        self.layer = layer
        self.in_degree = in_degree
        self.out_degree = []
        self.tensor_ids = tensor_ids  # nest struct
        self.arguments = arguments or {}
        for node in self.in_degree:
            if node:
                node.out_degree.append(self)

    def __repr__(self):
        return self.layer.name

    def __call__(self, *inputs):
        inputs = nest.map_structure(
            lambda x: inputs[x], nest_indices(self.tensor_ids))
        hooks = tensorlib._get_hooks()
        hooks = hooks.values()
        for hook in hooks:
            hook.before_forward(self.layer, inputs, self.arguments)
        with ops.name_scope(self.layer.name):
            outputs = self.layer.forward(*inputs, **self.arguments)
            # outputs return from __call__ should have same format
            # as from forward, e.g. outputs from forward is [?]
            unpack = not isinstance(outputs, (list, tuple))
            outputs = to_list(outputs)
        for hook in hooks:
            hook.after_forward(self.layer, outputs, inputs, self.arguments)
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
                 children,
                 out_nodes,
                 out_ids):
        self.network = network
        self.nodes = nodes
        self.children = children
        self.out_nodes = out_nodes
        self.out_ids = out_ids

    def __repr__(self):
        return str(self.nodes)

    def __call__(self, *inputs):
        # handle inputs according to its structure
        inputs = nest.flatten(inputs)
        log = {node: [inputs[i] for i in nest.flatten(node.tensor_ids)]
               for node in self.nodes[0]}
        for d_nodes in self.nodes[1:]:
            for node in d_nodes:
                log[node] = to_list(node(*[
                    log[node.in_degree[i]][idx] for i, idx in enumerate(
                        flatten_list(node.tensor_ids))]))
        # export outputs according to its structure
        outputs = [log[self.out_nodes[i]][idx]
                   for i, idx in enumerate(flatten_list(self.out_ids))]
        outputs = nest.map_structure(
            lambda x: outputs[x], nest_indices(self.out_ids))
        return unpack_singleton(outputs)


def build_graph_network(network, inputs, outputs):
    children = list()  # topological order
    nodes = set()
    # In order to maintain depth information of nodes,
    # we decide not to use topological structure to store nodes,
    # instead, we use a nest list to store nodes depth-wise
    nodes_by_depth = []
    out_nodes = [getattr(x, '_anchor')[0] for x in flatten_list(outputs)]
    out_ids = nest.map_structure(lambda x:  getattr(x, '_anchor')[1], outputs)
    for node in out_nodes:  # remove useless nodes in out_node's out_degree
        node.out_degree = list(set(node.out_degree) & set(out_nodes))
    cur_depth = []
    # Assume we already visited all current depth nodes' in-degree nodes
    for tensor in flatten_list(inputs):
        node = getattr(tensor, '_anchor')[0]
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
                        children=children,
                        out_nodes=out_nodes,
                        out_ids=out_ids)


# def build_graph_network(network, inputs, outputs):
#     cur_depth = []
#     for tensor in outputs:
#         node = getattr(tensor, '_anchor')[0]
#         node.out_degree.clear()
#         cur_depth.append(node)
#     stop_nodes = [getattr(tensor, '_anchor')[0] for tensor in inputs]
#     node_by_depth = []
#     nodes = set()
#     children = set()
#     depth = 0
#     while cur_depth:
#         next_depth = []
#         node_by_depth.append([])
#         for node in cur_depth:
#             if node in nodes or any(n not in nodes for n in node.out_degree):
#                 continue
#             node_by_depth[depth].append(node)
#             nodes.add(node)
#             children.add(node.layer)
#             if node in stop_nodes:
#                 continue
#             next_depth.extend(node.in_degree)
#         cur_depth = next_depth
#         depth += 1
#     return GraphNetwork(network=network,
#                         nodes=list(reversed(node_by_depth)),
#                         children=children)


def build_node(layer, inputs: (list, tuple), outputs: (list, tuple), arguments=None):
    in_degree = [getattr(tensor, '_anchor', (None, None))[0]
                 for tensor in flatten_list(inputs)]
    uses_lp = any(getattr(tensor, '_uses_learning_phase', False)
                  for tensor in flatten_list(inputs)) or has_arg(layer.forward, 'training')
    tensor_ids = nest.map_structure(lambda x: getattr(
        x, '_anchor', (None, None))[1], inputs)
    node = Node(layer=layer,
                in_degree=in_degree,
                tensor_ids=tensor_ids,
                arguments=arguments)
    for i, tensor in enumerate(flatten_list(outputs)):
        setattr(tensor, '_anchor', (node, i))
        setattr(tensor, '_uses_learning_phase', uses_lp)
