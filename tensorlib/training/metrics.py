from tensorlib.engine.base_layer import Layer
from tensorlib.engine import base_lib as F
from tensorlib.utils import to_list, valid_value, valid_range
from tensorlib import saving
from tensorflow.python.ops import variables
from tensorflow.python import ops
from tensorflow.python.summary import summary
from tensorflow.python.framework import ops as fops
from tensorflow.python.ops.confusion_matrix import confusion_matrix
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.util import nest

from enum import Enum
import numpy as np

NEG_INF = -1e10


class Reduction(Enum):
    SUM = 'sum'
    SUM_OVER_BATCH_SIZE = 'sum_over_batch_size'
    WEIGHT_MEAN = 'weighted_mean'


class Metric(Layer):

    def __init__(self,
                 name=None,
                 dtype=None,
                 **kwargs):
        super(Metric, self).__init__(name=name,
                                     dtype=dtype,
                                     **kwargs)
        self.built = True

    def reset_states(self):
        F.batch_set_value([(w, np.zeros(w.shape, w.dtype.as_numpy_dtype))
                           for w in self.weights])

    def forward(self, *args, **kwargs):
        pass

    def update_state(self, *args, **kwargs):
        raise NotImplementedError

    def result(self):
        raise NotImplementedError

    def add_summary_ops(self, name, value):
        with ops.name_scope(self.name):
            summary_op = summary.scalar(name=name, tensor=value)
            fops.add_to_collection(fops.GraphKeys.SUMMARIES, summary_op)

    def add_weight(self,
                   name,
                   shape=(),
                   dtype=None,
                   initial_value=None,
                   initializer=None,
                   synchronization=variables.VariableSynchronization.ON_READ,
                   aggregation=variables.VariableAggregation.SUM,
                   **kwargs):
        return super(Metric, self).add_weight(
            name=name,
            shape=shape,
            dtype=dtype,
            trainable=False,
            initializer=initializer,
            initial_value=initial_value,
            synchronization=synchronization,
            aggregation=aggregation)

    def __call__(self, *args, **kwargs):
        with ops.name_scope(self.name):
            updates = to_list(self.update_state(*args, **kwargs))
            with fops.control_dependencies(updates):
                result = self.result()
            # We are adding the metric object as metadata on every result tensor.
            # This metric instance will later be used to reset variable state after
            # each epoch of training.
            for res in to_list(nest.flatten(result)):
                setattr(res, '_metric_obj', self)
        return result


class Reduce(Metric):

    def __init__(self,
                 reduction,
                 name,
                 dtype=None):
        super(Reduce, self).__init__(name=name, dtype=dtype)
        self.reduction = valid_value(reduction, [
            Reduction.SUM_OVER_BATCH_SIZE,
            Reduction.WEIGHT_MEAN,
            Reduction.SUM])
        self.total = self.add_weight(name='total',
                                     initializer='zeros')
        if reduction in [Reduction.SUM_OVER_BATCH_SIZE,
                         Reduction.WEIGHT_MEAN]:
            self.count = self.add_weight(name='count',
                                         initializer='zeros')

    def update_state(self, values, sample_weight=None):
        values = math_ops.cast(values, self.dtype)
        if sample_weight is not None:
            sample_weight = math_ops.cast(sample_weight, self.dtype)
            values = math_ops.multiply(values, sample_weight)
        value_sum = math_ops.reduce_sum(values)
        with fops.control_dependencies([value_sum]):
            update_total_op = self.total.assign_add(value_sum)
        if self.reduction == Reduction.SUM:
            return update_total_op
        if self.reduction == Reduction.SUM_OVER_BATCH_SIZE:
            num_value = math_ops.cast(array_ops.size(values), self.dtype)
        else:
            if sample_weight is None:
                num_value = math_ops.cast(array_ops.size(values), self.dtype)
            else:
                num_value = math_ops.reduce_sum(sample_weight)
        with fops.control_dependencies([update_total_op]):
            return self.count.assign_add(num_value)

    def result(self):
        if self.reduction == Reduction.SUM:
            return array_ops.identity(self.total)
        else:
            return math_ops.div_no_nan(self.total, self.count)


class Mean(Reduce):

    def __init__(self,
                 name='mean',
                 dtype=None):
        super(Mean, self).__init__(
            reduction=Reduction.WEIGHT_MEAN,
            name=name, dtype=dtype)


class MeanMetricWrapper(Mean):

    def __init__(self,
                 function,
                 name=None,
                 dtype=None,
                 **kwargs):
        super(MeanMetricWrapper, self).__init__(name=name, dtype=dtype)
        self.function = function
        self.arguments = kwargs

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = math_ops.cast(y_true, self.dtype)
        y_pred = math_ops.cast(y_pred, self.dtype)
        matches = self.function(y_true, y_pred, **self.arguments)
        return super(MeanMetricWrapper, self).update_state(
            matches, sample_weight=sample_weight)


def _accuracy(y_true, y_pred):
    y_pred.shape.assert_is_compatible_with(y_true.shape)
    if y_true.dtype != y_pred.dtype:
        y_pred = math_ops.cast(y_pred, y_true.dtype)
    return F.float32(math_ops.equal(y_true, y_pred))


class Accuracy(MeanMetricWrapper):

    def __init__(self,
                 name='accuracy',
                 dtype=None):
        super(Accuracy, self).__init__(_accuracy, name=name, dtype=dtype)


def _binary_accuracy(y_true, y_pred, threshold=0.5):
    threshold = math_ops.cast(threshold, y_pred.dyte)
    y_pred = math_ops.cast(y_pred > threshold, y_pred.dyte)
    return math_ops.reduce_mean(math_ops.equal(y_true, y_pred), axis=-1)


class BinaryAccuracy(MeanMetricWrapper):

    def __init__(self,
                 name='binary_accuracy',
                 dtype=None):
        super(BinaryAccuracy, self).__init__(_binary_accuracy, name=name, dtype=dtype)


def _categorical_accuracy(y_true, y_pred):
    if F.ndim(y_true) == F.ndim(y_pred):
        y_true = math_ops.argmax(y_true, axis=-1)
    return F.float32(math_ops.equal(
        F.int64(y_true),
        math_ops.argmax(y_pred, axis=-1)))


class CategoricalAccuracy(MeanMetricWrapper):

    def __init__(self,
                 name='categorical_accuracy',
                 dtype=None):
        super(CategoricalAccuracy, self).__init__(_categorical_accuracy, name=name, dtype=dtype)


def _sparse_categorical_accuracy(y_true, y_pred):
    y_pred_rank = ops.convert_to_tensor(y_pred).shape.ndims
    y_true_rank = ops.convert_to_tensor(y_true).shape.ndims
    # If the shape of y_true is (num_samples, 1), squeeze to (num_samples,)
    if (y_true_rank is not None) and (y_pred_rank is not None) and (len(
            F.int_shape(y_true)) == len(F.int_shape(y_pred))):
        y_true = array_ops.squeeze(y_true, [-1])
    y_pred = math_ops.argmax(y_pred, axis=-1)

    # If the predicted output and actual output types don't match, force cast them
    # to match.
    if F.dtype(y_pred) != F.dtype(y_true):
        y_pred = math_ops.cast(y_pred, F.dtype(y_true))

    return F.float32(math_ops.equal(y_true, y_pred))


class SparseCategoricalAccuracy(MeanMetricWrapper):
    def __init__(self,
                 name='sparse_categorical_accuracy',
                 dtype=None):
        super(SparseCategoricalAccuracy, self).__init__(
            _sparse_categorical_accuracy,
            name=name, dtype=dtype)


class Precision(Metric):
    """
    precision = true_positive / (true_positive + false_positive)
    e.g. y_true = [0, 1, 1, 1], y_pre = [1, 0, 1, 1]
    precision = 2 / (2 + 1)
    """

    def __init__(self,
                 thresholds=None,
                 top_k=None,
                 class_id=None,
                 name=None,
                 dtype=None):
        super(Precision, self).__init__(name=name, dtype=dtype)
        self.init_thresholds = thresholds
        self.top_k = top_k
        self.class_id = class_id
        default_threshold = 0.5 if top_k else NEG_INF
        self.thresholds = [valid_range(th or default_threshold, (0, 1))
                           for th in to_list(thresholds)]
        self.true_positives = self.add_weight(
            name='true_positives',
            shape=(len(self.thresholds),),
            initializer='zeros')
        self.false_positives = self.add_weight(
            name='false_positives',
            shape=(len(self.thresholds),),
            initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # todo
        pass

    def reset_states(self):
        num_thresholds = len(self.thresholds)
        F.batch_set_value([
            (w, np.zeros((num_thresholds,))) for w in self.weights])

    def result(self):
        result = math_ops.div_no_nan(
            self.true_positives,
            self.true_positives + self.false_positives)
        return result[0] if len(self.thresholds) == 1 else result


class Recall(Precision):

    def __init__(self,
                 thresholds=None,
                 top_k=None,
                 class_id=None,
                 name=None,
                 dtype=None):
        super(Recall, self).__init__(
            thresholds=thresholds,
            top_k=top_k,
            class_id=class_id,
            name=name, dtype=dtype)


class MeanIou(Metric):
    def __init__(self, num_classes, name=None, dtype=dtypes.float64):
        super(MeanIou, self).__init__(name=name, dtype=dtype)
        self.total_value = self.add_weight(
            initial_value=array_ops.zeros(
                shape=[num_classes, num_classes], dtype=dtype),
            name='confusion_matrix',
            shape=[num_classes, num_classes],
            dtype=dtype)
        self.num_classes = F.int64(num_classes)
        self._mean_iou = None
        self._classes_iou = None

    def build(self, *args, **kwargs):
        pass

    def update_state(self, labels, predicts, weights=None):
        predicts = F.int64(predicts)
        labels = F.int64(labels)
        if labels.get_shape().ndims > 1:
            labels = array_ops.reshape(labels, [-1])
        if predicts.get_shape().ndims > 1:
            predicts = array_ops.reshape(predicts, [-1])
        if weights is not None and weights.get_shape().ndims > 1:
            weights = array_ops.reshape(weights, [-1])

        add_value = confusion_matrix(
            labels,
            predicts,
            num_classes=self.num_classes,
            weights=weights,
            dtype=dtypes.float64)
        with fops.control_dependencies([self.total_value]):
            total_update_op = state_ops.assign_add(self.total_value, add_value)

        sum_alone_row = F.float32(math_ops.reduce_sum(self.total_value, 0))
        sum_alone_col = F.float32(math_ops.reduce_sum(self.total_value, 1))

        diag = F.float32(array_ops.diag_part(self.total_value))
        denominator = sum_alone_row + sum_alone_col - diag
        valid_entries = math_ops.reduce_sum(F.float32(math_ops.not_equal(denominator, 0)))

        denominator = array_ops.where(
            math_ops.greater(denominator, 0),
            denominator,
            array_ops.ones_like(denominator))

        iou = math_ops.div(diag, denominator)
        self._mean_iou = array_ops.where(
            math_ops.greater(valid_entries, 0),
            math_ops.reduce_sum(iou) / valid_entries,
            0,
            name='mean_iou')
        self._classes_iou = array_ops.where(math_ops.not_equal(denominator, 0),
                                            iou,
                                            array_ops.zeros_like(denominator),
                                            name='classes_iou')
        return total_update_op

    def result(self):
        return array_ops.identity(self._mean_iou), array_ops.identity(self._classes_iou)


acc = accuracy = Accuracy
bce_acc = bce_accuracy = BinaryAccuracy
ce_acc = ce_accuracy = CategoricalAccuracy
sce_acc = sce_accuracy = SparseCategoricalAccuracy
precision = Precision
recall = Recall
mean_iou = MeanIou


def get(identifier):
    if identifier is None or callable(identifier):
        return identifier
    elif isinstance(identifier, str):
        if identifier in globals():
            return globals()[identifier]()
        else:
            raise ValueError("Can not find metric named `{}`"
                             " in module `metrics`".format(identifier))
    elif isinstance(identifier, dict):
        if 'class_name' not in identifier:
            raise ValueError("Identifier is illegal, ", str(identifier))
        return saving.load_dict(identifier)
    else:
        raise ValueError("Could not interpret identifier:", identifier)
