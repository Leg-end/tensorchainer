from tensorlib.engine.base_layer import Layer, param_tracker
from tensorlib.engine.base_lib import placeholder
from tensorlib.engine.graph_ops import build_node
from tensorflow.python.framework import tensor_shape

__all__ = ["InputLayer", "Input"]


class InputLayer(Layer):
    """
    This unique type of layer specifies the head of topological graph
    There have 2 situations for topological graph:
    1. param `input_tensor` is valid, that means connect two topological
    graphs, `input_tensor` is output from the last node of preceding graph

    2. parm `input_shape` or `batch_input_shape` is valid, that means building
    one topological graph uses this layer as head node and this layer's input
    as a placeholder to feed data
    e.g. >> output = Input(input_shape=(224, 224, 3))
         >> output = conv1(output)
         >> output = conv2(output)
         >> model = Network(inputs=Input.inputs, output=output)
         to use this model with new data
         >> sess.run(model.output[0], feed_dict={model.input[0]: new data})
    """
    @param_tracker()
    def __init__(self,
                 input_tensor=None,
                 input_shape=None,
                 batch_size=None,
                 batch_input_shape=None,
                 dtype=None,
                 name=None,
                 **kwargs):
        kwargs['trainable'] = False
        super(InputLayer, self).__init__(dtype=dtype, name=name, **kwargs)
        if input_tensor is None:
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
        elif not hasattr(input_tensor, '_anchor'):
            raise ValueError("To connect topological graphs, `input_tensor` must have attribute"
                             " `_anchor` to specify preceding node's meta data.")
        setattr(input_tensor, '_anchor', (None, 0))
        build_node(layer=self,
                   inputs=[input_tensor],
                   outputs=[input_tensor])
        self.inputs = [input_tensor]
        self._built = True

    def forward(self, inputs):
        return inputs


def Input(input_tensor=None,
          input_shape=None,
          batch_size=None,
          batch_input_shape=None,
          dtype=None,
          name='input'):
    input_layer = InputLayer(batch_input_shape=batch_input_shape,
                             dtype=dtype, name=name,
                             batch_size=batch_size,
                             input_shape=input_shape,
                             input_tensor=input_tensor)
    output = input_layer.inputs[0]
    return output
