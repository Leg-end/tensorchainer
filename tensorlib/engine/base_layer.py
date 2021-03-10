from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import typing as tp
import contextlib
import os
from collections import OrderedDict
from _collections_abc import MutableSequence

from tensorflow.python import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops as fops
from tensorflow.python.framework import errors_impl
from tensorflow.python.ops import variables
from tensorflow.python.util.function_utils import tf_inspect

from tensorlib import activation_ops
from tensorlib.engine import base_hook
from tensorlib.hooks import layer_hook
from tensorlib.engine import base_lib as F
from tensorlib.utils import nest
from tensorlib.utils.generic_util import unpack_singleton, validate_kwargs, to_list
from tensorlib.engine.name_manager import to_snake_case, get_unique_name
from tensorlib import initializers
from tensorlib import regularizers
import tensorlib

__all__ = ['Layer', 'param_tracker', 'LayerList']


def param_tracker(*skip_params):
    def decorator(method):
        @functools.wraps(method)
        def wrapper(*args, **kwargs):
            arg_spec = tf_inspect.getfullargspec(method)
            params = {}
            for i in range(1, len(args)):
                params[arg_spec.args[i]] = args[i]
            params.update(kwargs.pop('__config__', {}))
            params.update(kwargs)
            for param in skip_params:
                params.pop(param)
            kwargs['__config__'] = params
            return method(*args, **kwargs)

        return wrapper

    return decorator


class Layer(object):

    @property
    def trainable_weights(self):
        if self.trainable:
            for w in self._trainable_weights:
                yield w

    @property
    def non_trainable_weights(self):
        weights = self._non_trainable_weights if self.trainable\
            else self._non_trainable_weights + self._trainable_weights
        for w in weights:
            yield w

    @property
    def weights(self):
        for w in self._trainable_weights:
            yield w
        for w in self._non_trainable_weights:
            yield w

    @property
    def local_hooks(self):
        if not self._local_hooks:
            self._local_hooks = OrderedDict()
        return self._local_hooks

    @property
    def mirrors(self):
        return self._mirrors

    @property
    def input(self, index=0):
        return self._get_mirror_attribute_at(index, 'input_tensors')

    @property
    def output(self, index=0):
        return self._get_mirror_attribute_at(index, 'output_tensors')

    @property
    def built(self):
        return self._built

    @built.setter
    def built(self, value):
        self._built = value

    def __init__(self, activation=None, **kwargs):
        self._trainable_weights = []
        self._non_trainable_weights = []
        self._local_hooks = OrderedDict()
        self._built = False
        self._mirrors = []  # nodes presented as mirrors of this layer
        self.activation = activation_ops.get(activation)
        validate_kwargs(kwargs, {'dtype', 'name', '__config__', 'trainable'})
        # For serialization, recorded params in __init__
        self._config = kwargs.get('__config__', {})
        # self.name = kwargs.get('name', self.__class__.__name__)
        self._init_name_set(kwargs.get('name', None))
        self.trainable = kwargs.get('trainable', False)
        # Whether has updates have to update during inference
        # self.stateful = False
        dtype = kwargs.get('dtype', None)
        self.dtype = dtypes.float32 if dtype is None else dtypes.as_dtype(dtype)
        self.add_hook(layer_hook.LocalHook())

    def _init_name_set(self, name, zero_based=True):
        # if name != '':
        #     name = get_unique_name(
        #         to_snake_case(self.__class__.__name__
        #                       if name is None else name),
        #         zero_based=zero_based)
        # self.name = name
        # todo make the following codes works during save and load h5
        if name is None:
            name = get_unique_name(
                to_snake_case(self.__class__.__name__),
                zero_based=zero_based)
        # (Note: when name is '', no name scope will be used)
        # layers inside this layer will be flatten in graph
        self.name = name

    @contextlib.contextmanager
    def _name_scope(self):
        """
        Note: when name is '', no name scope will be used
        layers inside this layer will be flatten in graph
        """
        if self.name is '':
            yield
        else:
            with ops.name_scope(self.name) as scope:
                yield scope

    def _get_mirror_attribute_at(self, index, attr_name):
        if not self._mirrors:
            raise RuntimeError('The layer has never been called '
                               'and thus has no defined ' + attr_name + '.')
        if index > len(self._mirrors):
            raise ValueError('Asked to get ' + attr_name + ' at mirror ' +
                             str(index) + ', but the layer has only ' +
                             str(len(self._mirrors)) + ' mirrors.')
        return unpack_singleton(getattr(self._mirrors[index], attr_name))

    def check_define_before_run(self):
        if self.built:
            raise RuntimeError("Can only change layers inside %s before %s's `__call__`"
                               " or `build` was invoked, you should define before running." % (
                                self.name, self.name))

    def add_mirror(self, node):
        self._mirrors.append(node)

    def add_weight(self,
                   name,
                   shape=None,
                   dtype=None,
                   initial_value=None,
                   initializer=None,
                   regularizer=None,
                   trainable=None,
                   constraint=None,
                   **kwargs):
        """
        Add a variable weight to layer
        :param name: Name of weights
        :param shape: Shape of weights
        :param dtype: Data type of weights
        :param initial_value: Initial value of weights
        :param initializer: Initializer for weights
        :param regularizer: Regularizer for weights
        :param trainable: A boolean, whether the weight should
            be trained via backprop or not (assuming
            that the layer itself is also trainable).
        :param constraint: Optional constraint instance
        :return weight itself
        """
        dtype = dtype or self.dtype
        if initial_value is None:
            if shape is None:
                raise ValueError("When initial_value is not specified,"
                                 " shape for initializing must be specified.")
            if initializer is None:
                raise ValueError("When initial_value is not specified,"
                                 " initializer for initializing must be specified.")
            initial_value = initializers.get(initializer)(shape, dtype=dtype)
        synchronization = kwargs.get('synchronization', variables.VariableSynchronization.AUTO)
        if synchronization == variables.VariableSynchronization.ON_READ:
            if trainable:
                raise ValueError("Synchronization value can be set to"
                                 " VariableSynchronization.ON_READ only"
                                 " for non-trainable variables")
            else:
                trainable = False
        elif trainable is None:
            trainable = True
        weight = variables.Variable(initial_value=initial_value,
                                    trainable=trainable,
                                    dtype=dtype,
                                    constraint=constraint,
                                    name=name,
                                    **kwargs)
        if regularizer is not None:
            with ops.name_scope('weight_regularizer'):
                reg_loss = regularizers.get(regularizer)(weight)
                ops.add_to_collection(fops.GraphKeys.REGULARIZATION_LOSSES, reg_loss)
        if trainable:
            self._trainable_weights.append(weight)
        else:
            self._non_trainable_weights.append(weight)
        return weight

    def build(self, *args, **kwargs):
        self.built = True

    def forward(self, *inputs, **kwargs):
        raise NotImplementedError

    def __call__(self, *inputs, **kwargs):
        inputs = to_list(inputs)
        # self.local_hooks execute at last term
        hooks = OrderedDict(tensorlib._get_hooks(), **self.local_hooks)
        hooks = hooks.values()
        for hook in hooks:
            hook.before_forward(self, inputs, **kwargs)
        with self._name_scope():
            if not self.built:
                self.build(unpack_singleton(
                    nest.map_structure(F.int_shape, inputs)))
                self.built = True
            outputs = self.forward(*inputs, **kwargs)
            # outputs return from __call__ should have same format
            # as from forward, e.g. outputs from forward is [?]
            unpack = not isinstance(outputs, (list, tuple))
            outputs = to_list(outputs)
        for hook in hooks:
            hook.after_forward(self, outputs, inputs, **kwargs)
        if unpack:
            outputs = outputs[0]
        return outputs

    def layers(self, skip_self: bool = False):
        if not skip_self:
            yield self

    def children(self):
        if 0:
            yield

    def train(self):
        for layer in self.layers():
            layer.trainable = True

    def eval(self):
        for layer in self.layers():
            layer.trainable = False

    def count_params(self) -> int:
        pass

    def add_hook(self,
                 hook: base_hook.Hook,
                 name: tp.Optional[str] = None,
                 top=False):
        name = name or hook.name
        hooks = self.local_hooks
        if name in hooks:
            raise ValueError('Hook %s already existed' % name)
        if top:
            local_hooks = OrderedDict()
            local_hooks[name] = hook
            local_hooks.update(**hooks)
            self._local_hooks = local_hooks
        else:
            hooks[name] = hook
        hook.added(self)
        return self

    def delete_hook(self,
                    name: str):
        if name in self.local_hooks:
            self.local_hooks[name].deleted(self)
            del self.local_hooks[name]
        else:
            raise KeyError('Hook %s does not exist' % name)

    def load_weights(self, filepath, allow_skip=False, prefix=''):
        from tensorlib.saving.utils import load_hdf5_weights, load_ckpt_weights, is_hdf5_format
        if not os.path.exists(filepath):
            raise FileNotFoundError("File %s doesn't exist" % str(filepath))
        if is_hdf5_format(filepath):
            load_hdf5_weights(filepath=filepath,
                              lib_model=self,
                              allow_skip=allow_skip)
        else:
            try:
                load_ckpt_weights(filepath=filepath,
                                  prefix=prefix)
            except errors_impl.DataLossError:
                raise ValueError(
                    "File type must be 'hdf5', 'h5' or 'ckpt', but received: %s" % os.path.splitext(filepath)[-1])

    def save_weights(self, filepath):
        from tensorlib.saving.utils import save_hdf5_weights
        suffix = filepath[filepath.rfind('.') + 1:]
        if suffix.startswith("hdf5") or suffix.startswith("h5"):
            save_hdf5_weights(filepath=filepath, lib_model=self)
        elif suffix.startswith("ckpt"):
            pass
        else:
            raise ValueError(
                "File type must be 'hdf5' or 'ckpt', Other file type is not supported now.")


class LayerList(Layer, MutableSequence):

    @property
    def trainable_weights(self):
        for w in super(LayerList, self).trainable_weights:
            yield w
        for child in self.children():
            for w in child.trainable_weights:
                yield w

    @property
    def non_trainable_weights(self):
        for w in super(LayerList, self).non_trainable_weights:
            yield w
        for child in self.children():
            for w in child.non_trainable_weights:
                yield w

    @property
    def weights(self):
        for w in super(LayerList, self).weights:
            yield w
        for child in self.children():
            for w in child.weights:
                yield w

    def __init__(self, *layers: Layer, **kwargs):
        super(LayerList, self).__init__(**kwargs)
        self._children = []
        for layer in layers:
            if not isinstance(layer, Layer):
                raise TypeError("Expect type `Layer`, but"
                                " receive %s" % str(type(layer)))
            self.add_layer(layer)

    def __len__(self):
        return len(self._children)

    def __setattr__(self, name: str, value):
        if isinstance(value, Layer):
            raise TypeError('Can not register a new layer as an attribute, '
                            ' register layer by `insert`, `append` or `add_layer`')
        super(Layer, self).__setattr__(name, value)

    def __setitem__(self,
                    index: (int, slice),
                    value: (Layer, tp.Iterable[Layer])):
        self.check_define_before_run()
        self._children[index] = value

    def __getitem__(self, index):
        return self._children[index]

    def __delitem__(self, index: (int, slice)):
        self.check_define_before_run()
        del self._children[index]

    def __contains__(self, item):
        return item in self._children

    def __iter__(self):
        return iter(self._children)

    def __add__(self, other):
        if isinstance(other, type(self)):
            ret = other.__class__()
            for layer in self:
                ret.append(layer)
            for layer in other:
                ret.append(layer)
            return ret
        else:
            raise TypeError("unsupported operand type(s) for"
                            " +: '%s' and '%s'" % (
                             str(type(self)), str(type(other))))

    def __iadd__(self, other):
        self.check_define_before_run()
        if isinstance(other, type(self)):
            for layer in other:
                self.append(layer)
        else:
            raise TypeError("unsupported operand type(s) for"
                            " +: '%s' and '%s'" % (
                             str(type(self)), str(type(other))))

    def insert(self, index: int, layer: Layer):
        self.check_define_before_run()
        self._children.insert(index, layer)

    def layers(self, skip_self: bool = False):
        if not skip_self:
            yield self
        for child in self.children():
            for layer in child.layers():
                yield layer

    def children(self) -> tp.Iterator[Layer]:
        for child in self._children:
            yield child

    def add_layer(self, layer: Layer):
        self.append(layer)

    def get_layer(self, name=None, index=None):
        """
        Retrieves layer based on either its name,
         or index(in linear order), but index first
        :param name: str
        :param index: int
        :return: layer
        """
        if index is not None:
            assert index >= 0
            if index > len(self):
                raise ValueError("Requested to retrieve layer at index: " + str(index)
                                 + ", but model only has " + str(len(self)) + ' layers')
            else:
                return self[index]
        if not name:
            raise ValueError("Provide either name or index for retrieving")
        for layer in self.layers():
            if name == layer.name:
                return layer
        raise ValueError("No such layer: " + name)

    def remove_by_layer_type(self, type_name):
        [self.remove(layer) for layer in self
         if layer.__class__.__name__ == type_name]

    def count_by_layer_type(self, type_name):
        num = 0
        for layer in self:
            if layer.__class__.__name__ == type_name:
                num += 1
        return num

    def forward(self, *inputs):
        raise NotImplementedError
