from tensorflow.python import ops
from tensorflow.python.framework import ops as fops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import nn
from tensorflow.python.summary import summary
from tensorlib.engine.base_layer import Layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorlib.engine import base_lib as F
from tensorlib import saving
import numpy as np


epsilon = 1e-10
inf = np.inf


class Loss(Layer):
    def __init__(self,
                 name=None):
        super(Loss, self).__init__(name=name)
        self.summary_ops = []
        self.built = True

    def add_summary_ops(self, name, value):
        with ops.name_scope(self.name):
            summary_op = summary.scalar(name=name, tensor=value)
            self.summary_ops.append(summary_op)
            fops.add_to_collection(fops.GraphKeys.SUMMARIES, summary_op)

    def forward(self,  y_true, y_pred):
        raise NotImplementedError

    def __call__(self, y_true, y_pred, sample_weight=None):
        with ops.name_scope(self.name):
            losses = self.forward(y_true, y_pred)
            losses = math_ops.reduce_mean(losses)
            self.add_summary_ops(self.name + '_loss', losses)
        return losses


class MeanSquaredError(Loss):
    def forward(self,  y_true, y_pred):
        return math_ops.reduce_mean(
            math_ops.square(y_pred - y_true), axis=-1)


class MeanAbsoluteError(Loss):
    def forward(self,  y_true, y_pred):
        return math_ops.reduce_mean(
            math_ops.abs(y_pred - y_true), axis=-1)


class MeanAbsolutePercentageError(Loss):
    def forward(self,  y_true, y_pred):
        diff = math_ops.abs(
            (y_true - y_pred) / clip_ops.clip_by_value(
                t=math_ops.abs(y_true),
                clip_value_min=epsilon,
                clip_value_max=inf))
        return 100. * math_ops.reduce_mean(diff, axis=-1)


class MeanSquaredLogError(Loss):
    def forward(self,  y_true, y_pred):
        a = math_ops.log(clip_ops.clip_by_value(
            t=y_pred,
            clip_value_min=epsilon,
            clip_value_max=inf) + 1.)
        b = math_ops.log(clip_ops.clip_by_value(
            t=y_true,
            clip_value_min=epsilon,
            clip_value_max=inf) + 1.)
        return math_ops.reduce_mean(
            math_ops.square(a - b), axis=-1)


class SquaredHinge(Loss):
    def forward(self,  y_true, y_pred):
        return math_ops.reduce_mean(
            math_ops.square(math_ops.maximum(
                1. - y_true * y_pred, 0.)),
            axis=-1)


class Hinge(Loss):
    def forward(self,  y_true, y_pred):
        return math_ops.reduce_mean(
            math_ops.maximum(
                1. - y_true * y_pred, 0.),
            axis=-1)


class CategoricalHinge(Loss):
    def forward(self,  y_true, y_pred):
        pos = math_ops.reduce_sum(y_true * y_pred, axis=-1)
        neg = math_ops.reduce_max((1. - y_true) * y_pred, axis=-1)
        return math_ops.maximum(0., neg - pos + 1.)


class LogCosh(Loss):
    def forward(self,  y_true, y_pred):
        diff = y_pred - y_true
        value = diff + nn.softplus(-2. * diff) - math_ops.log(2.)
        return math_ops.reduce_mean(value, axis=-1)


class CategoricalCrossEntropy(Loss):
    def __init__(self,
                 from_logits=False,
                 axis=-1,
                 name=None):
        super(CategoricalCrossEntropy, self).__init__(name=name)
        self.from_logits = from_logits
        self.axis = axis

    def forward(self, y_true, y_pred):
        dims = list(range(len(y_pred.get_shape())))
        if self.axis != -1 and self.axis not in dims:
            raise ValueError("Axis out of y_pred's dimensions")
        if len(dims) - F.ndim(y_true) > 1:
            raise ValueError("y_pred's rank should be equal to y_true's"
                             " rank or y_true's rank + 1")
        elif len(dims) - F.ndim(y_true) == 1:
            y_true = array_ops.one_hot(y_true, depth=y_pred.shape[-1],
                                       dtype=y_pred.dtype)
        if not self.from_logits:
            if isinstance(y_pred, (ops.EagerTensor, variables.Variable)) \
                    or y_pred.op.type != 'Softmax':
                y_pred /= math_ops.reduce_sum(
                    y_pred, axis=self.axis, keepdims=True)
                y_pred = clip_ops.clip_by_value(
                    t=y_pred, clip_value_min=epsilon,
                    clip_value_max=1 - epsilon)
                return -math_ops.reduce_sum(
                    math_ops.cast(y_true, y_pred.dtype)
                    * math_ops.log(y_pred), axis=self.axis)
            else:
                # When softmax activation function is used for output operation, we
                # use logits from the softmax function directly to compute loss in order
                # to prevent collapsing zero when training.
                # See b/117284466
                assert len(y_pred.op.inputs) == 1
                y_pred = y_pred.op.inputs[0]
        return nn.softmax_cross_entropy_with_logits(labels=y_true,
                                                    logits=y_pred)


class SparseCategoricalCrossEntropy(Loss):
    def __init__(self,
                 from_logits=False,
                 axis=-1,
                 name=None):
        super(SparseCategoricalCrossEntropy, self).__init__(name=name)
        self.from_logits = from_logits
        self.axis = axis

    def forward(self,  y_true, y_pred):
        dims = list(range(len(y_pred.get_shape())))
        if self.axis != -1 and self.axis not in dims:
            raise ValueError("Axis out of y_pred's dimensions")
        if self.axis != -1 and self.axis != dims[-1]:
            perm = dims[:self.axis] + dims[self.axis + 1:]
            perm += [self.axis]
            y_pred = array_ops.transpose(y_pred, perm=perm)
        if not self.from_logits:
            if isinstance(y_pred, (ops.EagerTensor, variables.Variable))\
                    or y_pred.op.type != 'Softmax':
                y_pred = clip_ops.clip_by_value(
                    t=y_pred, clip_value_min=epsilon,
                    clip_value_max=1 - epsilon)
                y_pred = math_ops.log(y_pred)
            else:
                # When softmax activation function is used for output operation, we
                # use logits from the softmax function directly to compute loss in order
                # to prevent collapsing zero when training.
                # See b/117284466
                assert len(y_pred.op.inputs) == 1
                y_pred = y_pred.op.inputs[0]
        rank = len(y_pred.shape)
        self.axis = self.axis % rank
        if self.axis != rank - 1:
            permutation = list(range(self.axis)) + list(range(self.axis + 1, rank)) + [self.axis]
            y_pred = array_ops.transpose(y_pred, perm=permutation)
        shape = y_pred.shape
        y_true = F.int64(array_ops.reshape(y_true, [-1]))
        logits = array_ops.reshape(y_pred, [-1, int(shape[-1])])
        res = nn.sparse_softmax_cross_entropy_with_logits(
            labels=y_true,
            logits=logits)
        if len(shape) >= 3:
            return array_ops.reshape(res, array_ops.shape(y_pred)[:-1])
        else:
            return res


class BinaryCrossEntropy(Loss):
    def __init__(self,
                 from_logits=False,
                 name=None):
        super(BinaryCrossEntropy, self).__init__(name=name)
        self.from_logits = from_logits

    def forward(self,  y_true, y_pred):
        if not self.from_logits:
            y_pred = clip_ops.clip_by_value(
                t=y_pred, clip_value_min=epsilon,
                clip_value_max=1 - epsilon)
            y_pred = math_ops.log(y_pred / (1 - y_pred))
        return nn.sigmoid_cross_entropy_with_logits(labels=y_true,
                                                    logits=y_pred)


class KullbackLeiblerDivergence(Loss):
    def forward(self,  y_true, y_pred):
        y_true = clip_ops.clip_by_value(
            t=y_true, clip_value_min=epsilon,
            clip_value_max=1)
        y_pred = clip_ops.clip_by_value(
            t=y_pred, clip_value_min=epsilon,
            clip_value_max=1)
        return math_ops.reduce_sum(
            y_true * math_ops.log(y_true / y_pred), axis=-1)


class Poisson(Loss):
    def forward(self,  y_true, y_pred):
        return math_ops.reduce_mean(
            y_pred - y_true * math_ops.log(y_pred + epsilon),
            axis=-1)


class CosineProximity(Loss):
    def forward(self,  y_true, y_pred):
        y_true = nn.l2_normalize(y_true, axis=-1)
        y_pred = nn.l2_normalize(y_pred, axis=-1)
        return -math_ops.reduce_sum(y_true * y_pred, axis=-1)


mse = MSE = MeanSquaredError
mae = MAE = MeanAbsoluteError
mape = MAPE = MeanAbsolutePercentageError
msle = MSLE = MeanSquaredLogError
kld = KLD = KullbackLeiblerDivergence
cosine = CosineProximity
hinge = Hinge
square_hinge = SquaredHinge
categorical_hinge = CategoricalHinge
log_cosh = LogCosh
ce = CE = CategoricalCrossEntropy
bce = BCE = BinaryCrossEntropy
poisson = Poisson


def get(identifier):
    if identifier is None or callable(identifier):
        return identifier
    elif isinstance(identifier, str):
        if identifier in globals():
            return globals()[identifier]()
        else:
            raise ValueError("Can not find loss named `{}`"
                             " in module `losses`".format(identifier))
    elif isinstance(identifier, dict):
        if 'class_name' not in identifier:
            raise ValueError("Identifier is illegal, ", str(identifier))
        return saving.load_dict(identifier)
    else:
        raise ValueError("Could not interpret identifier:", identifier)
