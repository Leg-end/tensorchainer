from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorlib.engine.base_layer import Layer, param_tracker
from tensorlib.engine import base_lib as F
from tensorlib.engine.scope_manager import add_arg_scope
from tensorlib import initializers
from tensorlib import regularizers


__all__ = ["Dense"]


class Dense(Layer):
    @param_tracker()
    @add_arg_scope
    def __init__(self,
                 units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_normal',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 **kwargs):
        super(Dense, self).__init__(activation=activation, **kwargs)
        self.units = units
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = kernel_constraint
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_constraint = bias_constraint
        self.kernel = None
        self.bias = None

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        if input_dim is None:
            raise ValueError("Input dim must be defined legally,\
                     but received {}".format(str(input_dim)))
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(name='bias',
                                        shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        self._built = True

    def forward(self, inputs):
        rank = F.ndim(inputs)
        if rank > 2:
            outputs = math_ops.tensordot(
                inputs, self.kernel, [[rank - 1], [0]])
            outputs.set_shape(
                F.int_shape(inputs)[:-1] + (self.units,))
        else:
            outputs = math_ops.matmul(inputs, self.kernel)
        if self.use_bias:
            outputs = nn.bias_add(outputs, self.bias)
        if self.activation:
            outputs = self.activation(outputs)
        return outputs
