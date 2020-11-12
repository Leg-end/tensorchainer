from collections import Iterable
from abc import abstractmethod
from collections import OrderedDict
from tensorflow.python.framework import ops
import json
import random
import numpy as np
import numbers
import inspect
import itertools
import sys


class Aug(object):
    def __init__(self,
                 backend=None):
        self.backend = backend
        self.input_dtypes = []
        self._key_order = []

    @abstractmethod
    def augment(self, *args):
        raise NotImplementedError

    def transform(self, return_record=False, **kwargs):

        expected_keys = ["image", "segment", "heat_maps", "key_points", "b_boxes", "rhombus"]
        py_major, py_minor = sys.version_info[0], sys.version_info[1]
        is_py36_upper = py_major == 3 and py_minor >= 6
        if is_py36_upper:
            self._key_order = list(OrderedDict(kwargs).keys())
        elif not return_record and len(kwargs) >= 2:
            raise ValueError(
                "More than one outputs are only supported for 3.6+ "
                "as earlier python versions offer no way "
                "to retrieve the order of the provided named arguments.To "
                "still use more than two outputs, add 'return_record=True' as "
                "an argument and retrieve the outputs manually from the "
                "returned Record instance, e.g. via `record.image` to get augmented image.")

        assert any([key in kwargs for key in expected_keys]), (
            "Expected transform() to be called with one of the following "
            "named arguments: %s. Got none of theses." % (",".join(expected_keys),))

        unknown_args = [key for key in kwargs if key not in expected_keys]
        assert len(unknown_args) == 0, (
            "Got the following unknown keyword arguments in transform(): %s" %
            (", ".join(unknown_args)))

        try:
            fn_args = self.__class__.__dict__['augment'].__code__.co_varnames
            if "record" not in fn_args:
                raise TypeError("Not found argument `record` in augment()")
        except KeyError:
            raise NotImplementedError("augment not implemented")

        if self.backend == "tf":            # TODO
            assert all([_is_tf_array(arg) for arg in kwargs.values()]), (
                "Expected arguments of transform() with type tf.Tensor, but got %s"
                % (", ".join([type(arg).__name__ for arg in kwargs.values()])))
        else:
            assert all([_is_np_array(arg) for arg in kwargs.values()]), (
                    "Expected arguments of transform() with type tf.Tensor, but got %s"
                    % (", ".join([type(arg).__name__ for arg in kwargs.values()])))

        self.input_dtypes = [arg.dtype for arg in kwargs.values()]

        record = Record(
            image=kwargs.get("image", None),
            heat_maps=kwargs.get("heat_maps", None),
            segment=kwargs.get("segment", None),
            key_points=kwargs.get("key_points", None),
            b_boxes=kwargs.get("b_boxes", None))

        if not self._key_order:
            self._key_order = expected_keys
        record.order_keys = self._key_order

        record = self.augment(record)
        assert isinstance(record, Record), (
            "Expected Record output from %s.augment(), got type %s." % (
                self.__class__.__name__, type(record),))

        result = record.values
        return result

    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)

    def to_json(self):
        from tensorlib.saving import dump_object
        return json.dumps(dump_object(self), indent=2)

    def save_json(self, path):
        from tensorlib.saving import dump_object
        json.dump(dump_object(self), open(path, 'w'), indent=2)


class Sequential(Aug, list):
    def __init__(self, augmenters=None, backend=None):
        Aug.__init__(self, backend=backend)
        if augmenters is None:
            list.__init__(self, [])

        elif isinstance(augmenters, Aug):
            list.__init__(self, [augmenters])
        elif isinstance(augmenters, Iterable):
            assert all([isinstance(aug, Aug) for aug in augmenters]), (
                "Expected all `augments` to be Aug, got types %s." %
                (", ".join([str(type(aug)) for aug in augmenters])))
            list.__init__(self, augmenters)
        else:
            raise TypeError("Expected `None` or `Aug` or list of `Aug`, "
                            "got %s." % (type(augmenters),))

    def get_config(self):
        from tensorlib.saving import dump_object
        return {"augmenters": [dump_object(aug) for aug in self]}

    def augment(self, record):
        for index in range(len(self)):
            record = self[index].augment(record)
        return record


class SomeOf(Sequential):
    def __init__(self, num, augmenters=None):
        super(SomeOf, self).__init__(augmenters=augmenters)
        self._combinations = list(itertools.combinations(
            self, num))

    def augment(self, record):
        combination = random.choice(self._combinations)
        for aug in combination:
            record = aug.augment(record)
        return record


class OneOf(SomeOf):
    def __init__(self, augmenters=None):
        super(OneOf, self).__init__(
            num=1,
            augmenters=augmenters)


class Sometimes(Sequential):
    def __init__(self, augmenters=None):
        super(Sometimes, self).__init__(augmenters=augmenters)

    def augment(self, record):
        if random.randint(0, 1) == 0:
            for aug in self:
                record = aug.augment(record)
        return record


class Record(object):
    def __init__(self,
                 image=None,
                 segment=None,
                 heat_maps=None,
                 key_points=None,
                 b_boxes=None,
                 rhombus=None):
        self.image = image
        self.heat_maps = heat_maps
        self.key_points = key_points
        self.b_boxes = b_boxes
        self.segment = segment
        self.rhombus = rhombus

        self.order_keys = []
        self._get_init_args()

    @classmethod
    def _get_init_args(cls, skip=1):
        stack = inspect.stack()
        assert len(stack) >= skip + 1, (
            "The length of the inspection stack is shorter than the requested start position.")
        args, _, _, values = inspect.getargvalues(stack[skip][0])
        args = args[1:]
        del values["self"]
        if not (all([_is_np_array(values[arg]) for arg in args if values[arg] is not None]) or
           all([_is_tf_array(values[arg]) for arg in args if values[arg] is not None])):
            raise TypeError("Got list array")

    @property
    def values(self):
        _value = []
        for key in self.order_keys:
            if getattr(self, key) is not None:
                _value.append(getattr(self, key))
        if len(_value) == 1:
            return _value[0]
        return _value

    @property
    def dtypes(self):
        _type = []
        for key in self.order_keys:
            if getattr(self, key) is not None:
                _type.append(getattr(self, key).dtype)
        if len(_type) == 1:
            return _type[0]
        return _type


def _is_np_array(arr):
    return isinstance(arr, np.ndarray)


def _is_tf_array(arr):
    return isinstance(arr, ops.Tensor)


def _is_number(param):
    return isinstance(param, numbers.Integral) or isinstance(param, numbers.Real)


def check_number_param(param, name, list_or_tuple=False):
    if not list_or_tuple and _is_number(param):
        return param

    if isinstance(param, (tuple, list)):
        assert len(param) == 2, (
            "Expected parameter '%s' with type tuple of list to have exactly two "
            "entries, but got %d." % (name, len(param)))
        assert all([_is_number(v) for v in param]), (
            "Expected parameter '%s' with type tuple or list to only contain "
            "numbers, got %s." % (name, [type(v) for v in param],))
        return param
    raise Exception(
        "Expected %s, tuple of two numbers or single number, "
        "got %s." % (name, type(param)))


def _from_json(json_string):
    from tensorlib.saving import load_dict
    config = json.loads(json_string)
    return load_dict(config)


def load_json(path):
    with open(path, "r")as f:
        json_string = f.read()
    return _from_json(json_string)


def warning(msg, category=UserWarning, stack_level=2):
    """Generate a a warning with stacktrace.

    Parameters
    ----------
    msg : str
        The message of the warning.

    category : class
        The class of the warning to produce.

    stack_level : int, optional
        How many steps above this function to "jump" in the stacktrace when
        displaying file and line number of the error message.
        Usually ``2``.

    """
    import warnings
    warnings.warn(msg, category=category, stacklevel=stack_level)


if __name__ == '__main__':
    print(type(Sequential()))
