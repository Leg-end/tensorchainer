from tensorlib.data import backend as F
from tensorlib.utils import normalize_tuple
import random


class KeepAspectResizePoint(object):

    def __init__(self,
                 target_size,
                 random_offset=False):
        self.target_size = normalize_tuple(target_size, 2, 'target_size')
        self.random_offset = random_offset

    def __call__(self, points, image):
        return F.keep_aspect_resize_points(
            points[..., :2],
            image,
            self.target_size,
            random_seed=random.randint(1, 10e10)
            if self.random_offset else None)


class CropResizePoint(object):

    def __init__(self,
                 crop_size):
        self.crop_size = normalize_tuple(crop_size, 2, 'crop_size')

    def __call__(self, points, image):
        return F.crop_resize_points(
            points,
            image=image,
            crop_size=self.crop_size)
