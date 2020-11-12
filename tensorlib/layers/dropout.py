from tensorlib.engine.base_layer import Layer, param_tracker
from tensorlib.engine.scope_manager import add_arg_scope
from tensorlib.engine import base_lib as F
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import array_ops

__all__ = ["Dropout"]


class Dropout(Layer):
    @param_tracker()
    @add_arg_scope
    def __init__(self,
                 rate=0.5,
                 noise_shape=None,
                 seed=None,
                 **kwargs):
        super(Dropout, self).__init__(**kwargs)
        self.rate = min(1., max(0., rate))
        self.noise_shape = noise_shape
        self.seed = seed

    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape
        return nn_ops._get_noise_shape(inputs, self.noise_shape)

    def forward(self, inputs, training=None):
        if training is None:
            training = F.learning_phase()

        def dropped_inputs():
            return nn.dropout(
                inputs,
                noise_shape=self._get_noise_shape(inputs),
                seed=self.seed,
                rate=self.rate)
        outputs = F.smart_cond(training,
                               dropped_inputs,
                               lambda: array_ops.identity(inputs))
        return outputs
