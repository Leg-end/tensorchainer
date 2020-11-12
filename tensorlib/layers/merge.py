from tensorlib.engine.base_layer import Layer, param_tracker
from tensorlib import element_ops
from tensorlib.engine.scope_manager import add_arg_scope

__all__ = ["ElementWise", "Concat", "concat",
           "DimOp", "squeeze", "expand_dims"]


class ElementWise(Layer):
    @param_tracker()
    @add_arg_scope
    def __init__(self,
                 element_op,
                 activation=None,
                 **kwargs):
        if 'name' not in kwargs:
            if isinstance(element_op, str):
                kwargs['name'] = element_op
            elif callable(element_op):
                kwargs['name'] = element_op.__name__
        super(ElementWise, self).__init__(activation=activation, **kwargs)
        self.element_op = element_ops.get(element_op)

    def forward(self, *inputs):
        outputs = self.element_op(*inputs)
        if self.activation:
            outputs = self.activation(outputs)
        return outputs


class DimOp(Layer):
    @param_tracker()
    @add_arg_scope
    def __init__(self,
                 op,
                 axis=None,
                 activation=None,
                 **kwargs):
        super(DimOp, self).__init__(activation=activation, **kwargs)
        self.op = element_ops.get(op)
        self.axis = axis

    def forward(self, inputs):
        outputs = self.op(inputs, axis=self.axis)
        if self.activation:
            outputs = self.activation(outputs)
        return outputs


def squeeze(inputs, axis=None, **kwargs):
    return DimOp('squeeze', axis=axis,
                 name='squeeze', **kwargs)(inputs)


def expand_dims(inputs, axis=None, **kwargs):
    return DimOp('expand_dims', axis=axis,
                 name='expand_dims', **kwargs)(inputs)


class Concat(Layer):
    @param_tracker()
    @add_arg_scope
    def __init__(self,
                 axis=None,
                 activation=None,
                 **kwargs):
        super(Concat, self).__init__(activation=activation, **kwargs)
        self.concat = element_ops.get('concat')
        self.axis = axis

    def forward(self, inputs):
        outputs = self.concat(inputs, axis=self.axis)
        if self.activation:
            outputs = self.activation(outputs)
        return outputs


def concat(inputs, axis=None, **kwargs):
    return Concat(axis=axis, **kwargs)(inputs)
