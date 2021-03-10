from tensorflow.python.util import nest as tf_nest


def map_structure(func, *structure, inplace=False, **kwargs):
    """
    Same function as tensorflow.python.util.nest.map_structure
    but with tiny difference defined by `inplace`
    if inplace is False, totally same as tensorflow.python.util.nest.map_structure
    otherwise, results after `func` will be placed in `structure[0]`
    as original structure with no return term.
    """
    outputs = tf_nest.map_structure(func, *structure, **kwargs)
    if not inplace:
        return outputs
    else:
        for i, item in enumerate(outputs):
            structure[0][i] = item


def nest_indices(structure, start=0):
    indices = []
    for i, item in enumerate(structure):
        if isinstance(item, list):
            indices.append(nest_indices(item, start=len(indices)))
        else:
            indices.append(i + start)
    return indices


def index_to_structure(items, indices):
    return tf_nest.map_structure(lambda x: items[x], indices)


def flatten_list(structure):
    for item in structure:
        if isinstance(item, (list, tuple)):
            for x in flatten_list(item):
                yield x
        else:
            yield item


def flatten_dict(structure):
    for key, value in structure.items():
        if isinstance(value, dict):
            for x in flatten_dict(value):
                yield x
        else:
            yield key, value


def flatten(structure):
    return tf_nest.flatten(structure)
