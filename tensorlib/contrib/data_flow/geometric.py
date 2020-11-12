import numpy as np
from math import *
import cv2
import random
from tensorlib.contrib.data_flow.augmenter import Aug, check_number_param, _is_number


def get_t_matrix(points, t_points):
    matrix = np.zeros([8, 9])
    for i in range(0, 4):
        x = points[:, i].T
        y = t_points[:, i]

        matrix[i*2, 3:6] = -y[2] * x
        matrix[i*2, 6:] = y[1] * x
        matrix[i*2+1, :3] = y[2] * x
        matrix[i*2+1, 6:] = -y[0] * x
    _, _, v = np.linalg.svd(matrix)
    h = v[-1, :].reshape([3, 3])
    return h


def perspective_transform(h, w, zcop=1000., dpp=1000., angles=None):
    if angles is None:
        angles = np.asarray([0., 0., 0.])

    rads = np.deg2rad(angles)

    r_x = np.mat([[1, 0, 0], [0, cos(rads[0]), sin(rads[0])], [0, -sin(rads[0]), cos(rads[0])]])
    r_y = np.mat([[cos(rads[1]), 0, -sin(rads[1])], [0, 1, 0], [sin(rads[1]), 0, cos(rads[1])]])
    r_z = np.mat([[cos(rads[2]), sin(rads[2]), 0], [-sin(rads[2]), cos(rads[2]), 0], [0, 0, 1]])
    r = r_x * r_y * r_z

    xyz = np.mat([[0, 0, w, w], [0, h, 0, h], [0, 0, 0, 0]])
    hxy = np.mat([[0, 0, w, w], [0, h, 0, h], [1, 1, 1, 1]])
    xyz = xyz - np.mat([[w], [h], [0]]) / 2.
    xyz = r * xyz

    xyz = xyz - np.mat([[0], [0], [zcop]])
    h_xyz = np.concatenate([xyz, np.ones([1, 4])])
    p = np.mat([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1./dpp, 0]])
    _hxy = p * h_xyz
    _hxy = _hxy / _hxy[2, :]
    _hxy = _hxy + np.mat([[w], [h], [0]]) / 2.
    return get_t_matrix(points=hxy,  t_points=_hxy)


def image_project(image, t_matrix, target_h, target_w, border_value=0, flags=cv2.INTER_LINEAR):
    return cv2.warpPerspective(
        image,
        t_matrix,
        (target_w, target_h),
        borderValue=border_value,
        flags=flags)


def points_project(points, t_matrix):
    """
    arg:
        point: shape of (n, 2)
    return:
        result_points: shape of (n, 2)
    """
    if np.ndim(points) != 2:
        raise ValueError()
    points = points.transpose([1, 0])
    points = np.insert(points, 2, 1, axis=0).astype(np.float)
    result_points = np.matmul(t_matrix, np.mat(points))
    result_points = result_points / result_points[2]
    result_points = np.floor(result_points[:2].transpose([1, 0]))
    return np.asarray(result_points, dtype=np.int64)


def offset_transform(target_h, target_w, points, wh_ratio=.7):
    """
        point: shape of (2, n)
    """
    w_size = random.uniform(target_w * .7, target_w * 1.2)
    h_size = w_size / wh_ratio
    dw = random.uniform(0., target_w - w_size)
    dh = random.uniform(0., target_h - h_size)
    t_points = _get_rect_points(dw, dh, dw + w_size, dh + h_size)
    points = np.mat(np.insert(points, 2, 1, axis=0))
    t_matrix = get_t_matrix(points, t_points)
    return t_matrix


def _get_rect_points(tl_x, tl_y, br_x, br_y):
    return np.mat([[tl_x, br_x, br_x, tl_x],
                   [tl_y, tl_y, br_y, br_y],
                   [1.0, 1.0, 1.0, 1.0]])

# ####################################################################


def _t_matrix_offset_center(matrix, y, x):
    _x = (x - 1) / 2.0
    _y = (y - 1) / 2.0
    offset_matrix = np.array([[1, 0, _x], [0, 1, _y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -_x], [0, 1, -_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def _rotation_matrix(angle=(-30, 30), scale_range=(0.6, 1.1)):
    rads = np.pi / 180 * np.random.uniform(angle[0], angle[1])
    scale = np.random.uniform(scale_range[0], scale_range[1])
    r_matrix = np.array([[np.cos(rads), np.sin(rads), 0], [-np.sin(rads), np.cos(rads), 0], [0, 0, 1]])
    s_matrix = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]])
    return r_matrix.dot(s_matrix)


def _image_rotation(image, t_matrix, target_h, target_w, border_value=0, flags=cv2.INTER_LINEAR):
    return cv2.warpAffine(
        image, t_matrix[0:2, :],
        (target_w, target_h), flags=flags, borderValue=border_value)


def _point_rotation(points, t_matrix):
    if np.ndim(points) != 2:
        raise ValueError()
    points = np.asarray(points)
    points = points.transpose([1, 0])
    points = np.insert(points, 2, 1, axis=0)
    result_points = np.matmul(t_matrix, points)
    result_points = np.floor(result_points[0:2, :].transpose([1, 0]))
    return np.asarray(result_points, dtype=np.int64)


class Affine(Aug):
    def __init__(self,
                 target_size,
                 max_sum=None,
                 max_angle=None,
                 img_border_value=0,
                 seg_border_value=0,
                 backend=None):
        super(Affine, self).__init__(backend=backend)

        self.target_size = check_number_param(target_size, name="target_size", list_or_tuple=True)
        self.img_border_value = check_number_param(img_border_value, name="img_border_value")
        self.seg_border_value = check_number_param(seg_border_value, name="seg_border_value")
        self.max_angle = np.asarray(self._handle_max_angle(max_angle), dtype=np.float32)
        self.max_sum = float(max_sum) if max_sum is not None else 80.

    @classmethod
    def _handle_max_angle(cls, max_angle):
        if max_angle is None:
            return 40, 40, 10
        if isinstance(max_angle, (list, tuple)):
            assert len(max_angle) == 3, (
                    "Expected parameter '%s' with type tuple or list to have exactly three "
                    "entries, but got %d." % ("max_angle", len(max_angle)))
            assert all([_is_number(v) for v in max_angle]), (
                    "Expected parameter '%s' with type tuple or list to only contain "
                    "numbers, got %s." % ("max_angle", [type(v) for v in max_angle],))
            return max_angle
        elif isinstance(max_angle, dict):
            assert 'x' in max_angle or 'y' in max_angle or 'z' in max_angle, (
                "Expected max_angle dictionary to contain at "
                "least key \"x\" or key \"y\" or key \"z\". Found neither of them.")
            return tuple((check_number_param(max_angle.get('x', 0), name="max_angle['x']"),
                          check_number_param(max_angle.get('y', 0), name="max_angle['y']"),
                          check_number_param(max_angle.get('z', 0), name="max_angle['z']")))
        else:
            raise TypeError("Expected max_angle to be dictionary, but got %s"
                            % type(max_angle).__name__)

    def get_config(self):
        from tensorlib.saving import dump_iterable
        return {
            "target_size": dump_iterable(self.target_size),
            "max_angle": self.max_angle.tolist(),
            "max_sum": self.max_sum,
            "img_border_value": self.img_border_value,
            "seg_border_value": self.seg_border_value,
            "backend": self.backend}

    def augment(self, record):
        angles = np.random.rand(3) * self.max_angle

        if angles.sum() > self.max_sum:
            angles = (angles / angles.sum()) * (self.max_angle / self.max_angle.sum())

        target_h, target_w = self.target_size
        if record.image is None:
            raise ValueError("")

        h, w = record.image.shape[:2]
        image_coord = np.asarray([[0, w, w, 0], [0, 0, h, h]], dtype=float)
        t_matrix = offset_transform(target_h, target_w, image_coord)
        h_matrix = perspective_transform(target_h, target_w, angles=angles)
        affine_matrix = np.matmul(h_matrix, t_matrix)

        record.image = image_project(
            record.image,
            affine_matrix,
            target_h, target_w,
            border_value=self.img_border_value, flags=cv2.INTER_LINEAR)

        if record.segment is not None:
            record.segment = image_project(
                record.segment,
                affine_matrix,
                target_h, target_w,
                border_value=self.seg_border_value, flags=cv2.INTER_NEAREST)

        if record.key_points is not None:
            joints_list = []
            for joints in record.key_points:
                tag = np.asarray(joints)[..., 2:3]
                points = np.asarray(joints)[..., :2]
                points = points_project(points, affine_matrix)
                for i, point in enumerate(points.tolist()):
                    if point[0] < 0 or point[0] > target_w:
                        tag[i] = 0
                        continue
                    if point[1] < 0 or point[1] > target_h:
                        tag[i] = 0
                joints_list.append(np.concatenate([points, tag], axis=-1))
            record.key_points = np.asarray(joints_list)
        return record


class Rotation(Aug):
    def __init__(self,
                 target_size,
                 angle_range=None,
                 scale_range=None,
                 offset_center=True,
                 img_border_value=0,
                 seg_border_value=0,
                 backend=None):
        super(Rotation, self).__init__(backend=backend)
        self.angle_range = self._handle_angle(angle_range)
        self.scale_range = self._handle_scale(scale_range)
        self.target_size = check_number_param(
            target_size, name='target_size', list_or_tuple=True)

        self.offset_center = offset_center
        self.img_border_value = check_number_param(img_border_value, name="img_border_value")
        self.seg_border_value = check_number_param(seg_border_value, name="seg_border_value")

    @classmethod
    def _handle_angle(cls, angle_range):
        if angle_range is None:
            return -30, 30
        elif isinstance(angle_range, (tuple, list)):
            return check_number_param(angle_range, name="angle_range")
        elif isinstance(angle_range, dict):
            assert 'min' in angle_range or 'max' in angle_range, (
                "Expected angle_range dictionary to contain at "
                "least key \"min\" or key \"max\". Found neither of them.")
            return check_number_param(
                (angle_range.get('min', 0), angle_range.get('max', 0)),
                name="angle_range")
        else:
            raise TypeError("Expected angle_range to be dictionary, but got %s"
                            % type(angle_range).__name__)

    @classmethod
    def _handle_scale(cls, scale_range):
        if scale_range is None:
            return 0.8, 1.1
        elif isinstance(scale_range, (tuple, list)):
            return check_number_param(scale_range, name="scale_range")
        elif isinstance(scale_range, dict):
            assert 'min' in scale_range and 'max' in scale_range, (
                "Expected scale_range dictionary to contain at "
                "key \"min\" and key \"max\", but not Found.")
            return check_number_param((scale_range.get('min'), scale_range.get('max')),
                                      name="scale_range")
        else:
            raise TypeError("Expected scale_range to be dictionary, but got %s"
                            % type(scale_range).__name__)

    def augment(self, record):
        _h, _w = self.target_size

        t_matrix = _rotation_matrix(self.angle_range, self.scale_range)
        if self.offset_center:
            t_matrix = _t_matrix_offset_center(t_matrix, _h, _w)

        if record.image is not None:
            record.image = _image_rotation(
                record.image, t_matrix, _h, _w, border_value=self.img_border_value)

        if record.segment is not None:
            record.segment = _image_rotation(
                record.segment,
                t_matrix, _h, _w,
                border_value=self.seg_border_value, flags=cv2.INTER_NEAREST)

        if record.key_points is not None:
            joints_list = []
            for joints in record.key_points:
                tag = np.asarray(joints)[..., 2:3]
                points = np.asarray(joints)[..., :2]
                points = _point_rotation(points, t_matrix)
                for i, point in enumerate(points.tolist()):
                    if point[0] < 0 or point[0] > _w:
                        tag[i] = 0
                        continue
                    if point[1] < 0 or point[1] > _h:
                        tag[i] = 0
                joints_list.append(np.concatenate([points, tag], axis=-1))
            record.key_points = np.asarray(joints_list)
        return record
