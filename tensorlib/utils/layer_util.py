import numpy as np
from tensorlib.engine import int_shape
from tensorlib.utils import find_str


__all__ = ["count_params", "split_scope", "extract_layer_id"]


def count_params(params):
    if len(params) == 0:
        return 0
    return int(np.sum([np.prod(int_shape(p)) for p in set(params)]))


def split_scope(scope, layer_name):
    """
    Split name scope in layer's __call__ into context
     (i.e. a/b/c/layer.name=> a/b/c), layer name
     #Sitituation
     1. no context contain i.e. layer.name => returned context is '', name is layer.name
     2.layer_name not equal i.e. a/b/c/layer.name_i(this usually happen in reuse)
     we can still match layer.name in layer.name_i, won't be a problem
    """
    index = find_str(scope, layer_name, 0, reverse=True)
    context = scope[:max(index - 1, 0)]
    name = scope[index: -1]
    return context, name


def extract_layer_id(full_name, layer_name):
    """
    when layer_name is conv2d
    model/scope1/conv2d/bias_add:0 => scope1/conv2d
    or model/scope1/conv2d_1/bias_add:0 => scope1/conv2d_1
    or model/scope1/conv2d_1/forward_scope/bias_add:0 => scope1/conv2d_1
    """
    index = find_str(full_name, layer_name,
                     0, reverse=True, start=False)
    # In case layer was reused in same scope
    if full_name[index] != '/':
        index += full_name[index:].find('/')
    layer_id = full_name[full_name.find('/') + 1: index]
    return layer_id
