from tensorlib.engine.base_layer import Layer, param_tracker
from tensorlib.engine.base_lib import placeholder
from tensorlib.engine.graph_ops import build_node
from tensorflow.python.framework import tensor_shape

__all__ = ["InputLayer", "Input"]


class InputLayer(Layer):
    @param_tracker()
    def __init__(self,
                 input_shape=None,
                 batch_size=None,
                 batch_input_shape=None,
                 dtype=None,
                 name=None,
                 **kwargs):
        kwargs['trainable'] = False
        super(InputLayer, self).__init__(dtype=dtype, name=name, **kwargs)
        if input_shape and batch_input_shape:
            raise ValueError("Expect providing either input_shape or batch_input_shape,"
                             "can not be both.")
        if not batch_input_shape:
            if isinstance(input_shape, tensor_shape.TensorShape):
                input_shape = tuple(input_shape.as_list())
            elif isinstance(input_shape, int):
                input_shape = (input_shape,)
            batch_input_shape = (batch_size or 1,) + input_shape
        input_tensor = placeholder(shape=batch_input_shape,
                                   dtype=dtype, name=self.name)
        # input_tensor = ones(dtype=self.dtype)(batch_input_shape)
        setattr(input_tensor, '_anchor', (None, 0))
        build_node(layer=self,
                   inputs=[input_tensor],
                   outputs=[input_tensor])
        self.inputs = [input_tensor]
        self._built = True

    def forward(self, inputs):
        return inputs


def Input(input_shape=None,
          batch_size=None,
          batch_input_shape=None,
          dtype=None,
          name='input'):
    input_layer = InputLayer(batch_input_shape=batch_input_shape,
                             dtype=dtype, name=name,
                             batch_size=batch_size,
                             input_shape=input_shape)
    output = input_layer.inputs[0]
    return output
