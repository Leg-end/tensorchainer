from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python import ops
from tensorflow.python.framework.ops import control_dependencies
from tensorlib.engine import base_hook
from tensorlib.utils import nest
import sys


class NumericScaleHook(base_hook.Hook):
    name = 'NumericScaleHook'

    def after_forward(self, layer, outputs, inputs, **kwargs):
        message = ops.get_name_scope() + '/' + layer.name
        axes = list(range(len(outputs[0].shape) - 1))
        print_ops = []
        for i, x in enumerate(outputs):
            mean, var = nn.moments(x, axes=axes)
            print_ops.append(logging_ops.print_v2(array_ops.constant(
                message + '/output:%d' % i), output_stream=sys.stdout))
            print_ops.append(logging_ops.print_v2(mean, var, output_stream=sys.stdout))
        with control_dependencies(print_ops):
            for i, x in enumerate(outputs):
                outputs[i] = array_ops.identity(x)


class NumericHook(base_hook.Hook):
    name = 'NumericHook'

    def after_forward(self, layer, outputs, inputs, **kwargs):
        message = ops.get_name_scope() + '/' + layer.name
        nest.map_structure(lambda x: gen_array_ops.check_numerics(
            x, message=message), outputs, inplace=True)
