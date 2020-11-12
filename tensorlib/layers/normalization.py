from tensorflow.python.training import moving_averages
from tensorflow.python.framework import ops

from tensorlib.engine.base_layer import Layer, param_tracker
from tensorlib.utils.conv_util import normalize_data_format
from tensorlib.engine.scope_manager import add_arg_scope
from tensorlib.engine import base_lib as F
from tensorlib import initializers
from tensorlib import regularizers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import math_ops

__all__ = ['BatchNorm',
           'L2Norm', 'l2_norm', 'LRNorm', 'lrn']


class BatchNorm(Layer):
    """
    "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"
    (https://arxiv.org/abs/1502.03167)
    """

    @param_tracker()
    @add_arg_scope
    def __init__(self,
                 decay=0.99,
                 epsilon=1e-3,
                 activation=None,
                 data_format='channels_last',
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 moving_mean_initializer='zeros',
                 moving_var_initializer='ones',
                 gamma_regularizer=None,
                 beta_regularizer=None,
                 gamma_constraint=None,
                 beta_constraint=None,
                 **kwargs):
        super(BatchNorm, self).__init__(activation, **kwargs)
        self.data_format = normalize_data_format(data_format, 2)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_initializer = initializers.get(beta_initializer)
        self.moving_mean_initializer = initializers.get(moving_mean_initializer)
        self.moving_var_initializer = initializers.get(moving_var_initializer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_constraint = gamma_constraint
        self.beta_constraint = beta_constraint
        self.epsilon = epsilon
        self.decay = min(max(0.0, decay), 1.0)
        self.gamma = None
        self.beta = None
        self.moving_mean = None
        self.moving_var = None
        if self.data_format[-1] == "C":
            self.channel_axis = -1
        else:
            self.channel_axis = 1

    def build(self, input_shape=None):
        channels = input_shape[self.channel_axis]
        if channels is None:
            raise ValueError("Expect not none value from arg"
                             " 'input_shape', can not be both none")
        weight_shape = (channels,)
        self.gamma = self.add_weight(name='gamma',
                                     shape=weight_shape,
                                     initializer=self.gamma_initializer,
                                     regularizer=self.gamma_regularizer,
                                     constraint=self.gamma_constraint)
        self.beta = self.add_weight(name='beta',
                                    shape=weight_shape,
                                    initializer=self.beta_initializer,
                                    regularizer=self.beta_regularizer,
                                    constraint=self.beta_constraint)
        self.moving_mean = self.add_weight(name='moving_mean',
                                           shape=weight_shape,
                                           initializer=self.moving_mean_initializer,
                                           trainable=False)
        self.moving_var = self.add_weight(name='moving_var',
                                          shape=weight_shape,
                                          initializer=self.moving_var_initializer,
                                          trainable=False)

    def forward(self, inputs, training=None):
        if training is None:
            training = F.learning_phase()
        axes = [i for i in range(len(inputs.shape)) if i != self.channel_axis]
        mean, var = nn.moments(inputs, axes, keep_dims=False)

        def _train_fn():
            update_moving_mean = moving_averages.assign_moving_average(
                self.moving_mean, mean, self.decay, zero_debias=False)
            update_moving_var = moving_averages.assign_moving_average(
                self.moving_var, var, self.decay, zero_debias=False)
            with ops.control_dependencies([update_moving_mean, update_moving_var]):
                _outputs = F.batch_normalization(inputs, mean, var,
                                                 self.beta, self.gamma,
                                                 self.epsilon, self.data_format)
            return _outputs

        def _eval_fn():
            _outputs = F.batch_normalization(inputs, self.moving_mean, self.moving_var,
                                             self.beta, self.gamma,
                                             self.epsilon, self.data_format)
            return _outputs

        outputs = F.smart_cond(training,
                               true_fn=_train_fn,
                               false_fn=_eval_fn)
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs


class L2Norm(Layer):
    @param_tracker()
    @add_arg_scope
    def __init__(self,
                 scaling=False,
                 scale_initializer='ones',
                 data_format='channels_last',
                 epsilon=1e-12,
                 activation=None,
                 **kwargs):
        super(L2Norm, self).__init__(activation=activation, **kwargs)
        self.scaling = scaling
        self.scale_initializer = initializers.get(scale_initializer)
        self.data_format = normalize_data_format(data_format, 2)
        self.epsilon = epsilon
        self.norm_axis = None
        self.kernel = None

    def build(self, input_shape):
        if self.data_format[-1] == 'C':
            channel_axis = -1
        else:
            channel_axis = 1
        self.norm_axis = (channel_axis,)
        if self.scaling:
            kernel_shape = (input_shape[channel_axis],)
            self.kernel = self.add_weight(shape=kernel_shape,
                                          initializer=self.scale_initializer,
                                          name='scale',
                                          trainable=self.trainable)
            if channel_axis == 1:
                for _ in range(len(input_shape) - 2):
                    self.kernel = array_ops.expand_dims(self.kernel, axis=-1)

    def forward(self, inputs):
        outputs = nn.l2_normalize(
            x=inputs,
            axis=self.norm_axis,
            epsilon=self.epsilon,
            name='l2_norm')
        if self.scaling:
            outputs = math_ops.multiply(self.kernel, outputs)
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs


def l2_norm(inputs,
            scaling=False,
            scale_initializer='ones',
            data_format='channels_last',
            epsilon=1e-12):
    return L2Norm(scaling=scaling,
                  scale_initializer=scale_initializer,
                  data_format=data_format,
                  epsilon=epsilon)(inputs)


class LRNorm(Layer):
    @param_tracker()
    @add_arg_scope
    def __init__(self,
                 depth_radius=5,
                 bias=1.0,
                 alpha=0.0001,
                 beta=0.75,
                 activation=None,
                 **kwargs):
        super(LRNorm, self).__init__(activation=activation, **kwargs)
        self.depth_radius = depth_radius
        self.bias = bias
        self.alpha = alpha
        self.beta = beta

    def forward(self, inputs):
        outputs = nn.lrn(
            input=inputs,
            depth_radius=self.depth_radius,
            bias=self.bias,
            alpha=self.alpha,
            beta=self.beta,
            name='lrn_norm')
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs


def lrn(inputs,
        depth_radius=5,
        bias=1.0,
        alpha=0.0001,
        beta=0.75,
        **kwargs):
    return LRNorm(depth_radius=depth_radius,
                  bias=bias,
                  alpha=alpha,
                  beta=beta,
                  **kwargs)(inputs)
