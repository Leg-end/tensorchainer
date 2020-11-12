from tensorflow.python.util import nest


def nest_indices(structure, start=0):
    indices = []
    for i, item in enumerate(structure):
        if isinstance(item, list):
            indices.append(nest_indices(item, start=len(indices)))
        else:
            indices.append(i + start)
    return indices


def index_to_structure(items, indices):
    return nest.map_structure(lambda x: items[x], indices)


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
            yield value


"""if __name__ == '__main__':
    from tensorflow.python.util import nest
    from tensorlib.utils import flatten_list
    class Num:
        def __init__(self, v):
            self.v = v
        def __repr__(self):
            return str(self.v)
    a = [Num(1), Num(2), Num(3), [Num(4), Num(5)]]
    b = set(nest.flatten(a))
    for x in b:
        x.v = 1
    print(a)
    c = [(1, '1'), [(2, '2'), (3, '3')]]
    def fn(n):
        print(n)
        return n
    print(nest.map_structure(fn, c))


    def _struct_indices(ids, start=0):
        indices = []
        for i, idx in enumerate(ids):
            if isinstance(idx, list):
                indices.append(_struct_indices(idx, start=indices[-1]))
            else:
                indices.append(i + start)
        return indices
    print(_struct_indices([1, 2, [3, 4]]))"""
