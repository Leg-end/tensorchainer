from tensorflow.python.ops import nn
from tensorflow.python import ops
from tensorflow.python import clip_ops
from tensorlib import saving


def hard_sigmoid(x):
    x = 0.2 * x + 0.5
    zeros = ops.convert_to_tensor(0., dtype=x.dtype.base_dtype)
    ones = ops.convert_to_tensor(1., dtype=x.dtype.base_dtype)
    return clip_ops.clip_by_value(x, zeros, ones)


def relu(x):
    return nn.relu(x)


def relu6(x):
    return nn.relu6(x)


def tanh(x):
    return nn.tanh(x)


def sigmoid(x):
    return nn.sigmoid(x)


def leaky_relu(x, alpha=0.1):
    return nn.leaky_relu(x, alpha=alpha)


def softplus(x):
    return nn.softplus(x)


def softmax(x):
    return nn.softmax(x)


def linear(x):
    return x


def get(identifier):
    if identifier is None:
        return None
    elif isinstance(identifier, str):
        if identifier not in globals():
            raise ValueError("Can not find activation op named"
                             " `{}` in module `activation_ops`".format(identifier))
        else:
            return globals()[identifier]
    elif isinstance(identifier, dict):
        return saving.load_dict(identifier)
    elif callable(identifier):
        return identifier
    else:
        raise ValueError("Could not interpret identifier: ", str(identifier))
