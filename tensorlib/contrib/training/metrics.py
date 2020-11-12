from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorlib import engine
from tensorflow.python.ops.confusion_matrix import confusion_matrix
from tensorflow.python.ops import variable_scope
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import state_ops
from abc import abstractmethod
from tensorflow.python.framework import ops as fops
from tensorflow.python.ops import clip_ops
import numpy as np


def bbox_iou(predicts, labels):
    """
    shape [batch_size, 4]
    :param predicts: predict bounding box [x, y, w, h]
    :param labels: ground truth bounding box [x, y, w, h]
    """
    predicts = array_ops.unstack(predicts, axis=1)
    labels = array_ops.unstack(labels, axis=1)
    inter_w = predicts[2] + labels[2] - math_ops.maximum(
        predicts[0] + predicts[2], labels[0] + labels[2]) \
              + math_ops.minimum(predicts[0], labels[0])
    inter_h = predicts[3] + labels[3] - math_ops.maximum(
        predicts[1] + predicts[3], labels[1] + labels[3]) \
              + math_ops.minimum(predicts[1], labels[1])
    inter_w = math_ops.maximum(inter_w, array_ops.zeros_like(inter_w))
    inter_h = math_ops.maximum(inter_h, array_ops.zeros_like(inter_h))
    inter = inter_w * inter_h
    union = predicts[2] * predicts[3] + labels[2] * labels[3] - inter
    return math_ops.reduce_mean(inter / union)


def segment_iou(predicts, labels):
    """
    :param predicts: shape [batch_size, h, w, c]
    :param labels: shape [batch_size, h, w]
    """
    num_classes = predicts.shape[-1]
    labels = array_ops.one_hot(labels, depth=num_classes)
    predicts = engine.bool(predicts)
    labels = engine.bool(labels)
    inter = engine.float32(math_ops.logical_and(predicts, labels))
    union = engine.float32(math_ops.logical_or(predicts, labels))
    inter = math_ops.reduce_sum(inter, axis=[0, 1, 2])
    union = math_ops.reduce_sum(union, axis=[0, 1, 2])
    classes_iou = inter / union
    mean_iou = math_ops.reduce_mean(classes_iou)
    return mean_iou, classes_iou


def keypoint_iou(predicts, labels, fall_factors, mask=None):
    """
    fall_factors order in ['nose', ‘neck’， 'left_eye', 'right_eye', 'left_ear',
     'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow',
      'right_elbow', 'left_wrist', 'right_wrist', 'left_hip',
      'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
    :param predicts: [batch_size, num_joints, 2]
    :param labels: [batch_size, num_joints, 3]
    :param fall_factors: Constants for joints, calculated by the group of researchers from COCO.
    :param mask: [batch_size, h, w] mask of person
    """
    # fall_factors = array_ops.constant([[
    #     0.026, 0.025, 0.025, 0.035,
    #     0.035, 0.079, 0.079, 0.072, 0.072,
    #     0.062, 0.062, 0.107, 0.107, 0.087,
    #     0.087, 0.089, 0.089]])
    fall_factors = math_ops.square(fall_factors) * 2.
    num = len(fall_factors.shape)
    if num == 1:
        fall_factors = array_ops.reshape(fall_factors, [1, -1])
    elif num != 2:
        raise ValueError("Expect `fall_factors` has rank 1 or 2,"
                         " but received: " + str(num))
    gt_joints = labels[:, :, :2]
    gt_flags = labels[:, :, 2]
    gt_flags = array_ops.where(math_ops.greater(gt_flags, 0),
                               array_ops.ones_like(gt_flags),
                               array_ops.zeros_like(gt_flags))
    distance = math_ops.reduce_sum(math_ops.square(gt_joints - predicts), axis=2)
    iou = -distance / fall_factors
    if mask is not None:
        mask = engine.float32(mask)
        iou /= (math_ops.reduce_sum(mask) / math_ops.reduce_prod(mask.shape))

    iou = math_ops.reduce_sum(math_ops.exp(iou) * gt_flags)
    iou /= math_ops.reduce_sum(gt_flags)
    return iou


class Metric(engine.Layer):
    def __init__(self, name=None, dtype=None, **kwargs):
        super(Metric, self).__init__(name=name, dtype=dtype, **kwargs)
        self._built = True
        self.update_ops = []

    def forward(self, *args, **kwargs):
        pass

    def add_weight(self,
                   name,
                   shape=None,
                   dtype=None,
                   initial_value=None,
                   initializer=None,
                   synchronization=variable_scope.VariableSynchronization.ON_READ,
                   aggregation=variable_scope.VariableAggregation.SUM,
                   **kwargs):
        return super(Metric, self).add_weight(
            name=name,
            shape=shape,
            dtype=dtype,
            trainable=False,
            initial_value=initial_value,
            initializer=initializer,
            synchronization=synchronization,
            aggregation=aggregation,
            **kwargs)

    @abstractmethod
    def update_state(self, *args, **kwargs):
        raise NotImplementedError('Must be implemented in subclasses.')

    @abstractmethod
    def result(self):
        raise NotImplementedError('Must be implemented in subclasses.')

    def __call__(self, *args, **kwargs):
        self.update_state(*args, **kwargs)
        with fops.control_dependencies(self.update_ops):
            result = self.result()
        for update_op in self.update_ops:
            fops.add_to_collection(fops.GraphKeys.UPDATE_OPS, update_op)
        return result


class SegmentIOU(Metric):
    def __init__(self, num_classes, name=None, dtype=dtypes.float64):
        super(SegmentIOU, self).__init__(name=name, dtype=dtype)
        self.total_value = self.add_weight(
            initial_value=array_ops.zeros(shape=[num_classes, num_classes], dtype=dtype),
            name='confusion_matrix',
            shape=[num_classes, num_classes],
            dtype=dtype)
        self.num_classes = engine.int64(num_classes)
        self._mean_iou = None
        self._classes_iou = None

    def build(self, *args, **kwargs):
        pass

    def update_state(self, labels, predicts, weights=None):
        predicts = engine.int64(predicts)
        labels = engine.int64(labels)
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
            self.update_ops.append(state_ops.assign_add(self.total_value, add_value))

        sum_alone_row = engine.float32(math_ops.reduce_sum(self.total_value, 0))
        sum_alone_col = engine.float32(math_ops.reduce_sum(self.total_value, 1))

        diag = engine.float32(array_ops.diag_part(self.total_value))
        denominator = sum_alone_row + sum_alone_col - diag
        valid_entries = math_ops.reduce_sum(engine.float32(math_ops.not_equal(denominator, 0)))

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

    def result(self):
        return self._mean_iou, self._classes_iou


def operator(distance):
    fall_factors = array_ops.constant([[
        0.026, 0.079, 0.025, 0.025, 0.035,
        0.035, 0.079, 0.079, 0.072, 0.072,
        0.062, 0.062, 0.107, 0.107, 0.087,
        0.087, 0.089, 0.089]])
    return math_ops.exp(-distance / math_ops.square(1.) / 2.)  # math_ops.square(fall_factors) * 2.


class SingleObject(Metric):
    def __init__(self, num_joint, output_stride=1, name=None, dtype=dtypes.float32):
        super(SingleObject, self).__init__(name=name, dtype=dtype)
        self.num_joint = num_joint
        self.output_stride = output_stride
        self.total_value = self.add_weight(
            initial_value=array_ops.zeros(shape=[num_joint]),
            name='vector',
            shape=[num_joint],
            dtype=dtype)
        self.count = self.add_weight(
            initial_value=array_ops.zeros(shape=[num_joint]),
            name='count',
            shape=[num_joint],
            dtype=dtype)

    def update_state(self, labels, predictions, weights=None):
        """
        Arg:
            labels: shape of (#batch, #num_joint, 3) : num_joint=18
            predictions: shape of (#batch, #height, #width, #num_joint) : num_joint=18
            weights: shape of (#batch, #height*output_stride, #width*output_stride)
        """
        b, h, w, c = predictions.get_shape()
        euclid_dists = []
        masks = []
        predictions = array_ops.reshape(array_ops.transpose(predictions, (0, 3, 1, 2)), (-1, h * w))
        # weights = image.resize(weights, (h, w), align_corners=True,
        #                        method=image.ResizeMethod.NEAREST_NEIGHBOR)
        a_ = []
        labels = array_ops.reshape(labels, (-1, 3))
        if weights is None:
            weights = array_ops.ones_like(predictions)
        else:
            weights = array_ops.tile(array_ops.expand_dims(weights, axis=-1), [1, 1, 1, c])
            weights = array_ops.reshape(array_ops.transpose(weights, (0, 3, 1, 2)), (-1, h * w))

        for prediction, label, weight in zip(array_ops.unstack(predictions),
                                             array_ops.unstack(labels),
                                             array_ops.unstack(weights)):
            prediction = array_ops.where(math_ops.equal(weight, 1), prediction,
                                         array_ops.ones_like(prediction)
                                         * array_ops.constant(np.nan, dtype=dtypes.float32))
            arg_index = math_ops.arg_max(prediction, 0)
            masks.append(clip_ops.clip_by_value(label[2], 0., 1.)
                         * array_ops.gather(weight, arg_index))
            col, row = arg_index % w, math_ops.floor_div(arg_index, w)
            arg_coord = math_ops.cast(array_ops.stack([col, row]), dtype=dtypes.float32)
            # arg_coord = (self.output_stride / 2 - 0.5) + self.output_stride * arg_coord
            a_.append(arg_coord)

            euclid_dists.append(math_ops.reduce_sum((arg_coord - label[:2]) ** 2))

        masks = array_ops.reshape(array_ops.stack(masks), (b, c))
        euclid_dists = array_ops.reshape(array_ops.stack(euclid_dists), (b, c))
        self.a = array_ops.reshape(array_ops.stack(a_), (b, c, 2))

        with fops.control_dependencies([self.count]):
            self.update_ops.append(state_ops.assign_add(
                self.count, math_ops.reduce_sum(masks, axis=0)))

        add_value = operator(euclid_dists) * masks

        with fops.control_dependencies([self.total_value]):
            self.update_ops.append(state_ops.assign_add(
                self.total_value, math_ops.reduce_sum(add_value, axis=0)))

    def result(self):
        return math_ops.div_no_nan(self.total_value, self.count)


class BBoxIOU(Metric):

    def __init__(self, name='bbox_metric', dtype=dtypes.float32):
        super(BBoxIOU, self).__init__(name=name, dtype=dtype)
        self.iou = self.add_weight(
            initial_value=array_ops.zeros(shape=[], dtype=dtype),
            name='iou',
            shape=[],
            dtype=dtype)

    def update_state(self, predicts, labels):
        iou = bbox_iou(predicts, labels)
        with fops.control_dependencies([self.iou]):
            self.update_ops.append(state_ops.assign(self.iou, iou))

    def result(self):
        return self.iou
