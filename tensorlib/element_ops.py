from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorlib.engine.base_lib import dot
from tensorlib import saving


def average(*inputs):
    assert len(inputs) > 0
    output = inputs[0]
    for i in range(1, len(inputs)):
        output += inputs[i]
    return output / len(inputs)


def maximum(*inputs):
    assert len(inputs) > 0
    output = inputs[0]
    for i in range(1, len(inputs)):
        output = math_ops.maximum(output, inputs[i])
    return output


def minimum(*inputs):
    assert len(inputs) > 0
    output = inputs[0]
    for i in range(1, len(inputs)):
        output = math_ops.minimum(output, inputs[i])
    return output


OPS = {
    "add": math_ops.add,
    "sub": math_ops.subtract,
    "multiply": math_ops.multiply,
    "divide": math_ops.divide,
    "average": average,
    "maximum": maximum,
    "minimum": minimum,
    "dot": dot,
    "concat": array_ops.concat,
    "squeeze": array_ops.squeeze,
    "expand_dims": array_ops.expand_dims
}


def get(identifier):
    if identifier is None:
        return identifier
    elif isinstance(identifier, str):
        if identifier not in OPS:
            raise ValueError("Can not find element op op named"
                             " `{}` in module `element_ops`".format(identifier))
        else:
            return OPS[identifier]
    elif isinstance(identifier, dict):
        return saving.load_dict(identifier)
    elif callable(identifier):
        return identifier
    else:
        raise ValueError("Could not interpret identifier: ", str(identifier))
