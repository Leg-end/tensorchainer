from tensorflow.python.ops import math_ops
from tensorlib import saving
import numpy as np


class Regularizer:
    """Regularizer base class.
    """

    def __init__(self, name=None):
        self.name = name or self.__class__.__name__

    def __repr__(self):
        return self.name

    def __call__(self, x):
        return 0.

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_config(self):
        return {'name': self.name}


class L1L2(Regularizer):
    """Regularizer for L1 and L2 regularization.

    # Arguments
        l1: Float; L1 regularization factor.
        l2: Float; L2 regularization factor.
    """

    def __init__(self, l1=0., l2=0., name=None):
        super(L1L2, self).__init__(name=name)
        self.l1 = np.asarray(l1, 'float32')
        self.l2 = np.asarray(l2, 'float32')

    def __call__(self, x):
        regularization = 0.
        if self.l1:
            regularization += math_ops.reduce_sum(self.l1 * math_ops.abs(x))
        if self.l2:
            regularization += math_ops.reduce_sum(self.l2 * math_ops.square(x))
        return regularization

    def get_config(self):
        config = super(L1L2, self).get_config().copy()
        config.update({'l1': float(self.l1),
                       'l2': float(self.l2)})
        return config


def l1(l=0.01):
    return L1L2(l1=l, name='l1')


def l2(l=0.01):
    return L1L2(l2=l, name='l2')


def l1_l2(l1=0.01, l2=0.01):
    return L1L2(l1=l1, l2=l2, name='l1_l2')


def get(identifier):
    if identifier is None or callable(identifier):
        return identifier
    elif isinstance(identifier, str):
        if identifier in globals():
            return globals()[identifier]()
        else:
            raise ValueError("Can not find regularizer named `{}`"
                             " in module `regularizers`".format(identifier))
    elif isinstance(identifier, dict):
        if 'class_name' not in identifier:
            raise ValueError("Identifier is illegal, ", str(identifier))
        return saving.load_dict(identifier)
    else:
        raise ValueError("Could not interpret identifier:", identifier)
