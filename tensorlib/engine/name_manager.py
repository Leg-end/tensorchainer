from collections import defaultdict
from tensorflow.python.framework import ops
import weakref
import re

__all__ = ["to_snake_case", "get_unique_name",
           "reset_default_graph_uid",
           "get_default_graph_uid"]

_GRAPH_UID_DICTS = weakref.WeakKeyDictionary()


def get_default_graph_uid():
    global _GRAPH_UID_DICTS
    graph = ops.get_default_graph()
    if graph not in _GRAPH_UID_DICTS:
        _GRAPH_UID_DICTS[graph] = defaultdict(int)
    return _GRAPH_UID_DICTS[graph]


def reset_default_graph_uid():
    global _GRAPH_UID_DICTS
    _GRAPH_UID_DICTS = weakref.WeakKeyDictionary()


def get_unique_name(name,
                    graph_uid=None,
                    avoid_names=None,
                    zero_based=False):
    if graph_uid is None:
        graph_uid = get_default_graph_uid()

    if avoid_names is None:
        avoid_names = set()

    proposed_name = None
    while proposed_name is None or proposed_name in avoid_names:
        number = graph_uid[name]
        if zero_based:
            if number:
                proposed_name = name + '_' + str(number)
            else:
                proposed_name = name
        elif number:
            proposed_name = name + '_' + str(number)
        else:
            graph_uid[name] += 1
            proposed_name = name + '_' + str(graph_uid[name])
        graph_uid[name] += 1
    return proposed_name


def to_snake_case(name):
    intermediate = re.sub('(.)([A-Z][a-z0-9]+)', r'\1_\2', name)
    insecure = re.sub('([a-z])([A-Z])', r'\1_\2', intermediate).lower()
    # If the class is private the name starts with "_" which is not secure
    # for creating scopes. We prefix the name with "private" in this case.
    if insecure[0] != '_':
        return insecure
    return 'private' + insecure
