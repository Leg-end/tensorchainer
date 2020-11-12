import numpy as np
import math
import random
from tensorlib.contrib.data_flow.image_ops import np_image_ops as ops
from tensorlib.contrib.data_flow.augmenter import Aug, check_number_param


def _keep_aspect_resize_points(points, raw_size, target_size, random_seed=None, is_limit=True):
    raw_h, raw_w = raw_size
    target_h, target_w = target_size
    raw_aspect = raw_w / raw_h
    target_aspect = target_w / target_h
    if raw_aspect > target_aspect:
        h = math.floor(target_w / raw_aspect)
        if random_seed:
            random.seed(random_seed)
            padding = random.randint(0, target_h - h)
        else:
            padding = (target_h - h) / 2

        ys = points[..., 1:2] * (h / raw_h)
        if is_limit:
            xs = np.minimum(target_w, np.maximum(0., points[..., 0:1] * target_w / raw_w))
            ys = padding + np.minimum(target_h, np.maximum(0., ys))
        else:
            xs = points[..., 0:1] * target_w / raw_w
            ys = padding + ys

    else:
        w = math.floor(target_h * raw_aspect)
        if random_seed:
            random.seed(random_seed)
            padding = random.randint(0, target_w - w)
        else:
            padding = (target_w - w) / 2
        xs = points[..., 0:1] * (w / raw_w)
        if is_limit:
            xs = padding + np.minimum(target_w, np.maximum(0., xs))
            ys = np.minimum(target_h, np.maximum(0., points[..., 1:2] * target_h / raw_h))
        else:
            xs = padding + xs
            ys = points[..., 1:2] * target_h / raw_h
    result_points = np.floor(np.concatenate([xs, ys], axis=-1))
    return np.asarray(result_points, dtype=np.int64)


class KeepAspectResize(Aug):
    def __init__(self,
                 target_size,
                 random_offset=False,
                 img_border_value=0,
                 seg_border_value=0,
                 backend=None):
        super(KeepAspectResize, self).__init__(backend=backend)
        self.target_size = check_number_param(target_size, name="target_size", list_or_tuple=True)
        self.random_offset = random_offset
        self.img_border_value = check_number_param(img_border_value, name="img_border_value")
        self.seg_border_value = check_number_param(seg_border_value, name="sef_border_value")

    def augment(self, record):
        t_h, t_w = self.target_size
        if record.image is None:
            raise ValueError("")

        r_h, r_w = record.image.shape[:2]
        random_seed = None
        if self.random_offset:
            random_seed = random.randint(0, 10e10)

        record.image = ops.keep_aspect_resize_padding(
            record.image, t_h, t_w,
            random_seed=random_seed,
            border_value=self.img_border_value)

        if record.segment is not None:
            record.segment = ops.keep_aspect_resize_padding(
                record.segment, t_h, t_w,
                random_seed=random_seed,
                interp=ops.ResizeMethod.NEAREST,
                border_value=self.seg_border_value)

        if record.key_points is not None:
            result_joints = _keep_aspect_resize_points(
                record.key_points[..., :2],
                (r_h, r_w),
                self.target_size,
                random_seed=random_seed)
            record.key_points = np.concatenate([result_joints, record.key_points[..., 2:]], axis=-1)

        return record


def _resize_points(points, raw_size, target_size):
    r_h, r_w = raw_size
    t_h, t_w = target_size
    ratio_h = t_h / r_h
    ratio_w = t_w / r_w

    flat_points = points.reshape(-1, 3)
    result_points = []
    for point in flat_points:
        if point[0] < 0. or point[1] < 0.:
            result_points.append([0, 0, 0])
            continue
        point = [int(point[0] * ratio_w + 0.5), int(point[1] * ratio_h + 0.5), int(point[2])]
        if point[0] > t_w - 1 or point[1] > t_h - 1:
            result_points.append([0, 0, 0])
            continue
        result_points.append(point)
    result_points = np.floor(np.reshape(result_points, points.shape))
    return np.asarray(result_points, dtype=np.int64)


class ResizeAndCrop(Aug):
    def __init__(self,
                 crop_size,
                 backend=None):
        super(ResizeAndCrop, self).__init__(backend=backend)
        self.crop_size = check_number_param(crop_size, name="crop_size", list_or_tuple=True)

    def augment(self, record):

        crop_h, crop_w = self.crop_size
        if record.image is None:
            raise ValueError("")

        input_h, input_w, _ = record.image.shape
        vertical_ratio = crop_h / input_h
        horizontal_ratio = crop_w / input_w
        rescale_ratio = max(vertical_ratio, horizontal_ratio)

        rescale_size = (int(round(input_h * rescale_ratio)), int(round(input_w * rescale_ratio)))

        if rescale_size[0] > crop_h:
            crop_range_h = np.random.randint(0, rescale_size[0] - crop_h)
            record.image = ops.resize(
                record.image,
                rescale_size)[crop_range_h:crop_range_h + crop_h, :, :]

            if record.key_points is not None:
                record.key_points = _resize_points(
                    record.key_points,
                    (input_h, input_w),
                    rescale_size)
                key_points = []
                for point in record.key_points.reshape(-1, 3):
                    if (point[1] >= crop_range_h) and (point[1] <= crop_range_h + crop_h - 1):
                        point = [int(point[0]), int(point[1] - crop_range_h), int(point[2])]
                    else:
                        point = [0, 0, 0]
                    key_points.append(point)
                record.key_points = np.reshape(key_points, record.key_points.shape)

            if record.segment is not None:
                record.segment = ops.resize(
                    record.segment,
                    rescale_size,
                    interp=ops.ResizeMethod.NEAREST)[crop_range_h:crop_range_h + crop_h, :]

        elif rescale_size[1] > crop_w:
            crop_range_w = np.random.randint(0, rescale_size[1] - crop_w)
            record.image = ops.resize(
                record.image,
                rescale_size)[:, crop_range_w:crop_range_w + crop_w, :]

            if record.key_points is not None:
                record.key_points = _resize_points(
                    record.key_points,
                    (input_h, input_w),
                    rescale_size)

                key_points = []
                for point in record.key_points.reshape(-1, 3):

                    if (point[0] >= crop_range_w) and (point[0] <= crop_range_w + crop_w - 1):
                        point = [int(point[0] - crop_range_w), int(point[1]), int(point[2])]
                    else:
                        point = [0, 0, 0]
                    key_points.append(point)
                record.key_points = np.reshape(key_points, record.key_points.shape)

            if record.segment is not None:
                record.segment = ops.resize(
                    record.segment,
                    rescale_size,
                    interp=ops.ResizeMethod.NEAREST)[:, crop_range_w:crop_range_w + crop_w]
        else:
            pass
        return record
