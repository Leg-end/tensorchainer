from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import marshal
import codecs
import types as python_types
from six import string_types, integer_types
import warnings
import sys
import os
import binascii
import json


BASE_TYPES = (bool, float, type(None)) + string_types + integer_types
_CACHE = {}


def _load_module(module_name, name):
    global _CACHE
    key = (module_name, name)
    if key in _CACHE:
        return _CACHE[key]
    module_parts = module_name.split(".")
    if sys.version_info[0] == 3 and module_parts[0] == "__builtin__":
        module_parts = ["six", "moves", "builtins"] + module_parts[1:]
    value = None
    for i in range(1, len(module_parts) + 1):
        try:
            qualified_name = ".".join(module_parts[:i])
            value = __import__(
                qualified_name,
                fromlist=module_parts[:i - 1])
        except ImportError:
            break
    if value is None:
        raise ImportError(module_parts[0])
    for attr_name in module_parts[i:] + name.split("."):
        value = getattr(value, attr_name)
    _CACHE[key] = value
    return value


def dump_fn(func):
    if func.__name__ == '<lambda>':
        raw_code = marshal.dumps(getattr(func, '__code__'))
        if os.name == 'nt':
            raw_code = raw_code.replace(b'\\', b'/')
        code = codecs.encode(raw_code, 'base64').decode('ascii')
        defaults = getattr(func, '__defaults__')
        if getattr(func, '__closure__') is not None:
            closure = [c.cell_contents for c in getattr(func, '__closure__')]
        else:
            closure = None
        return {'__module__': func.__module__, '__function__': [
            code, closure, defaults], '__function_type__': 'lambda'}
    else:
        return {'__module__': func.__module__, '__function__': func.__name__,
                '__function_type__': 'function'}


def load_fn(code_meta, custom_module=None):
    if custom_module is not None:
        if not isinstance(custom_module, dict):
            raise TypeError("`custom_module` must be a dict")
        global _CACHE
        _CACHE.update(**custom_module)
    func_meta = code_meta['__function__']
    function_type = code_meta['__function_type__']
    if function_type == 'lambda':
        code = func_meta[0]
        closure = func_meta[1]
        defaults = func_meta[2]

        def ensure_value_to_cell(value):
            """
            Wrap value with class `cell`
            """

            def dummy_fn():
                value

            cell_value = dummy_fn.__closure__[0]
            if not isinstance(value, type(cell_value)):
                return cell_value
            else:
                return value

        if closure is not None:
            closure = tuple(ensure_value_to_cell(value) for value in closure)
        try:
            raw_code = codecs.decode(code.encode('ascii'), 'base64')
        except (UnicodeEncodeError, binascii.Error):
            raw_code = code.encode('raw_unicode_escape')
        code = marshal.loads(raw_code)
        return python_types.FunctionType(
            code=code, globals=_CACHE,
            name=code.co_name, argdefs=defaults,
            closure=closure)
    elif function_type == 'function':
        return _load_module(code_meta['__module__'], func_meta)
    else:
        raise TypeError("Unknown function type: ", function_type)


def dump_attrs(obj):
    try:
        init_code = obj.__init__.__func__.__code__
    except AssertionError:
        try:
            init_code = obj.__new__.__func__.__code__
        except AssertionError:
            raise ValueError("Cannot determine args to %s.__init__" % (obj,))
    attr_names = init_code.co_varnames[:init_code.co_argcount][1:]
    serialized_meta = {}
    for name in attr_names:
        if not hasattr(obj, name):
            raise ValueError("Missing necessary attribute `{}` in Object `{}`".format(
                name, obj.__class__.__name__))
        serialized_meta[name] = dump(getattr(obj, name))
    return serialized_meta


def dump_dict(meta):
    # dummy key `__tmp__` to store keys in meta which have special type
    serialized_meta = {'__tmp__': []}
    for key, value in meta.items():
        if isinstance(key, BASE_TYPES):
            serialized_meta[key] = dump(value)
        else:
            serialized_meta['__tmp__'].append([dump(key), dump(value)])
    if len(serialized_meta['__tmp__']) == 0:
        serialized_meta.pop('__tmp__')
    return serialized_meta


def dump_iterable(meta):
    serialized_meta = [dump(value) for value in meta]
    if isinstance(meta, (tuple, set)):  # json doesn't have tuple, set
        serialized_meta = dict({'__value__': serialized_meta}, **dump_class(type(meta)))
    return serialized_meta


def dump_object(meta):
    name = meta.__class__.__name__
    if hasattr(meta, 'get_config'):
        config = meta.get_config()
        base_types = BASE_TYPES + (dict, list)
        for k, v in config.items():
            if not isinstance(v, base_types):
                raise TypeError("Config from {} must be serializable to json format"
                                ", but received un-serializable element with type {},"
                                " value {}.\n For more efficient serialization, Object"
                                " implements method `get_config` should do serialization"
                                " manually in method `get_config` by using tensorlib.saving."
                                "\n You may forgot to do serialization in {}'s `get_config`.".format(
                                 name, type(v), str(v), name))
    else:
        warnings.warn("Sub class of Object {} should implement method `get_config`"
                      " to maintain its own parameters needed in `__init__`. "
                      "Missing that property may occur exception when doing "
                      "deserialization.".format(name))
        try:
            config = dump_attrs(meta)
        except ValueError:
            raise ValueError("Failed to serialize object `{}` with value {}".format(
                              name, str(meta)))
    return dict({'__config__': config}, **dump_class(type(meta)))


def dump_class(meta):
    return {"__module__": meta.__module__, "__class__": meta.__name__}


def load_dict(meta, custom_module=None):
    if custom_module is not None:
        if not isinstance(custom_module, dict):
            raise TypeError("`custom_module` must be a dict")
        global _CACHE
        _CACHE.update(custom_module)
    if '__class__' in meta:
        cls = _load_module(meta['__module__'], meta['__class__'])
        if '__value__' in meta:  # (tuple, set)
            return cls(load(meta['__value__']))
        elif '__config__' in meta:  # object
            config = load(meta['__config__'])
            if hasattr(cls, 'from_config'):
                return cls.from_config(config)
            else:
                return cls(**config)
        else:  # class
            return cls
    elif '__function__' in meta:  # function
        return load_fn(meta)
    else:  # dict
        deserialized_meta = {}
        if '__tmp__' in meta:
            for value in meta.pop('__tmp__'):
                deserialized_meta[load(value[0])] = load(value[1])
        for key, value in meta.items():
            deserialized_meta[load(key)] = load(value)
        return deserialized_meta


def dump(meta):
    meta_type = type(meta)
    if meta_type in BASE_TYPES:
        return meta
    elif meta_type is dict:
        return dump_dict(meta)
    elif meta_type in (list, set, tuple):
        return dump_iterable(meta)
    elif meta_type in (python_types.FunctionType,
                       python_types.BuiltinFunctionType,
                       python_types.LambdaType):
        return dump_fn(meta)
    elif meta_type is type:
        return dump_class(meta)
    elif isinstance(meta, object):
        return dump_object(meta)
    else:
        raise ValueError("Unknown type with: " + str(meta_type))


def load(meta, custom_module=None):
    if custom_module is not None:
        if not isinstance(custom_module, dict):
            raise TypeError("`custom_module` must be a dict")
        global _CACHE
        _CACHE.update(**custom_module)
    if isinstance(meta, BASE_TYPES):
        return meta
    elif isinstance(meta, dict):
        return load_dict(meta)
    elif isinstance(meta, list):
        return [load(value) for value in meta]
    else:
        raise ValueError("Unknown type with: " + str(type(meta)))


def from_json(json_str, custom_module=None):
    return load_dict(json.loads(json_str), custom_module=custom_module)


def from_json_file(path, custom_module=None):
    if not os.path.exists(path):
        raise ValueError("Can not find path: " + path)
    return load_dict(json.load(open(path)), custom_module=custom_module)


def to_json(meta):
    return json.dumps(dump(meta), indent=2)


def to_json_file(meta, path):
    json.dump(dump(meta), open(path, 'w'), indent=2)
