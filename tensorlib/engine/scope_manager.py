from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_decorator
from types import FunctionType
import re


__all__ = ['add_arg_scope', 'arg_scope']

_ARGSTACK = [{}]  # {_key_op: {arg: value},...}
_DECORATED_OPS = {}  # {_key_op: (arg1, arg2, ...)}


def _get_arg_stack():
    if _ARGSTACK:
        return _ARGSTACK
    else:
        _ARGSTACK.append({})
        return _ARGSTACK


def _name_op(op):
    return op.__module__, op.__name__


# update 2020/3/12.
# Before this, this function only be used in `__init__` method, used in other method
# will wrong and BUG in `return func.__code__.co_varnames[-kwargs_length + flag: func.__code__.co_argcount]`
# After the update, any function can be supported, By adding
# `func.__code__.co_varnames[-kwargs_length:func.__code__.co_argcount]`
def _kwarg_names(func):
    # __defaults__ get all defaults keyword value
    if func.__defaults__:
        kwargs_length = len(func.__defaults__)
        flag = 0
    else:
        kwargs_length = 0    # func.__defaults__ else 0
        flag = 1

    # __code__.co_varnames return tuple include function keyword name
    # __code__.co_argcount get number of arg of function
    # like >>> def fn(a, b=1, c=2): pass; will get >>> ('a', 'b', 'c')
    if func.__name__ == "__init__":
        return func.__code__.co_varnames[-kwargs_length + flag: func.__code__.co_argcount] # BUG if func is not __init__
    else:
        return func.__code__.co_varnames[-kwargs_length:func.__code__.co_argcount]  # for any method, besides __init__


def arg_scope_key(op):
    """
    Arg:
        1. op is python function object:
            example: use @add_arg_scope on `__init__` of  class Conv2D, then
                str(op) will get '<function Conv2D.__init__ at 0x00000258CB3E2BF8>'

        2. op either be function like __init__ and user Custom functions or name of class which inherit the
        layers.core.Layer class
    """
    # the key which class name
    if re.findall("__init__", str(op)):
        # key will be class name such `Conv2D`
        key = re.sub("^.*(\\s)", "", re.sub("(\\.).*$", "", str(op)))        # NOT BUG MAY BE
        return getattr(op, '_key_op', key)
    else:
        return getattr(op, '_key_op', str(op))


def _add_op(op):
    key_op = arg_scope_key(op)
    _DECORATED_OPS[key_op] = _kwarg_names(op)


def current_arg_scope():
    stack = _get_arg_stack()
    return stack[-1]


def has_arg_scope(func):
    return arg_scope_key(func) in _DECORATED_OPS


@tf_contextlib.contextmanager
def arg_scope(list_ops, **kwargs):
    if isinstance(list_ops, dict):
        if kwargs:
            raise ValueError("When attempting to reuse a scope by supplying a"
                             "dictionary, kwargs must be empty")
        current_scope = list_ops.copy()
        try:
            _get_arg_stack().append(current_scope)
            yield current_scope
        finally:
            _get_arg_stack().pop()

    else:
        if not isinstance(list_ops, list):
            raise TypeError("list_ops must be a list or reused scope")
        try:
            current_scope = current_arg_scope().copy()
            for op in list_ops:
                if isinstance(op, FunctionType):  # update in 2020/3/12
                    key = arg_scope_key(op)
                    if not has_arg_scope(op):
                        raise ValueError("% is not decorated with @add_arg_scope",
                                         _name_op(op))
                else:
                    key = arg_scope_key(op.__name__)
                    if not has_arg_scope(op.__name__):
                        raise ValueError("% is not decorated with @add_arg_scope",
                                         _name_op(op))
                if key in current_scope:
                    current_kwargs = current_scope[key].copy()
                    current_kwargs.update(kwargs)
                    current_scope[key] = current_kwargs
                else:
                    current_scope[key] = kwargs.copy()
            _get_arg_stack().append(current_scope)
            yield current_scope
        finally:
            _get_arg_stack().pop()


def add_arg_scope(func):
    """
    Arg : Python function object
    """
    def func_with_args(*args, **kwargs):
        current_scope = current_arg_scope()
        current_args = kwargs
        key_func = arg_scope_key(func)
        if key_func in current_scope:
            current_args = current_scope[key_func].copy()
            current_args.update(kwargs)
        return func(*args, **current_args)
    _add_op(func)
    setattr(func_with_args, '_key_op', arg_scope_key(func))
    return tf_decorator.make_decorator(func, func_with_args)
