from _collections_abc import MutableSequence
from tensorlib.utils import normalize_tuple
from tensorlib.data import backend as F
import itertools
import typing as tp
import random


__all__ = ['KeepAspectResize', 'CropResize', 'RandomBrightness',
           'RandomContrast', 'RandomFlip', 'RandomHue',
           'Compose', 'SomeOf', 'OneOf', 'Sometimes', 'RandomOrder',
           'Affine', 'Rotation']


class KeepAspectResize(object):
    def __init__(self,
                 target_size,
                 method=F.ResizeMethod.BILINEAR,
                 random_offset=False,
                 border_value=0):
        self.target_size = normalize_tuple(target_size, 2, 'target_size')
        self.method = method
        self.random_offset = random_offset
        self.border_value = border_value

    def __call__(self, inputs):
        h, w = self.target_size
        inputs = F.keep_aspect_resize_padding(
            inputs, h, w,
            random_seed=random.randint(1, 10e10)
            if self.random_offset else None,
            border_value=self.border_value,
            method=self.method)
        return inputs


class CropResize(object):

    def __init__(self,
                 crop_size,
                 method=F.ResizeMethod.BILINEAR):
        self.crop_size = normalize_tuple(crop_size, 2, 'crop_size')
        self.method = method

    def __call__(self, inputs):
        return F.crop_resize(inputs, self.crop_size, method=self.method)


class RandomBrightness(object):

    def __init__(self, prob, delta=32.):
        self.prob = min(1., max(0., prob))
        self.delta = delta

    def __call__(self, inputs):
        return F.random_brightness(inputs, self.prob, self.delta)


class RandomHue(object):
    def __init__(self, prob, delta, channel_format="RGB"):
        super(RandomHue, self).__init__()
        self.prob = min(1., max(0., prob))
        self.delta = delta
        assert channel_format in ['RGB', 'BGR']
        self.channel_format = channel_format

    def __call__(self, inputs):
        return F.random_hue(
            inputs, self.prob, self.delta,
            channel_format=self.channel_format)


class RandomContrast(object):

    def __init__(self, prob, lower=0.6, upper=1.2):
        self.prob = min(1., max(0., prob))
        assert upper > lower
        self.lower = lower
        self.upper = upper

    def __call__(self, inputs):
        return F.random_contrast(
            inputs, self.prob, self.lower, self.upper)


class RandomFlip(object):

    def __init__(self, prob, flip_code):
        self.prob = min(1., max(0., prob))
        assert flip_code in [0, 1]
        self.flip_code = flip_code

    def __call__(self, inputs):
        return F.random_flip(inputs, self.prob, self.flip_code)


class Affine(object):
    def __init__(self,
                 target_size,
                 max_sum=None,
                 max_angle=None,
                 border_value=0,
                 method=F.ResizeMethod.BILINEAR):
        self.target_size = normalize_tuple(target_size, 2, 'target_size')
        self.max_sum = float(max_sum) if max_sum is not None else 80.
        self.max_angle = normalize_tuple(max_angle, 3, 'max_angle')\
            if max_angle is not None else (40, 40, 10)
        self.border_value = border_value
        self.method = method

    def __call__(self, inputs):
        return F.random_affine(inputs, size=self.target_size,
                               max_sum=self.max_sum,
                               max_angle=self.max_angle,
                               border_value=self.border_value,
                               method=self.method)


class Rotation(object):
    def __init__(self,
                 target_size,
                 angle_range=None,
                 scale_range=None,
                 offset_center=True,
                 border_value=0,
                 method=F.ResizeMethod.BILINEAR):
        self.target_size = normalize_tuple(target_size, 2, 'target_size')
        self.angle_range = angle_range if angle_range is not None else (-30, 30)
        self.scale_range = scale_range if scale_range is not None else (0.8, 1.1)
        self.offset_center = offset_center
        self.border_value = border_value
        self.method = method

    def __call__(self, inputs):
        return F.random_rotation(inputs, size=self.target_size,
                                 angle_range=self.angle_range,
                                 scale_range=self.scale_range,
                                 offset_center=self.offset_center,
                                 border_value=self.border_value,
                                 method=self.method)


class Compose(MutableSequence):

    def __init__(self, *transforms):
        self.transforms = []
        self.transforms.extend(transforms)

    def __len__(self):
        return len(self.transforms)

    def __setitem__(self,
                    index: (int, slice),
                    value: (callable, tp.Iterable[callable])):
        self.transforms[index] = value

    def __getitem__(self, index):
        return self.transforms[index]

    def __delitem__(self, index: (int, slice)):
        del self.transforms[index]

    def __contains__(self, item):
        return item in self.transforms

    def __iter__(self):
        return iter(self.transforms)

    def __add__(self, other):
        if isinstance(other, type(self)):
            ret = Compose()
            for t in self:
                ret.append(t)
            for t in other:
                ret.append(t)
            return ret
        else:
            raise TypeError("unsupported operand type(s) for"
                            " +: '%s' and '%s'" % (
                             str(type(self)), str(type(other))))

    def __iadd__(self, other):
        if isinstance(other, type(self)):
            for layer in other:
                self.append(layer)
        else:
            raise TypeError("unsupported operand type(s) for"
                            " +: '%s' and '%s'" % (
                             str(type(self)), str(type(other))))

    def insert(self, index: int, item: callable):
        self.transforms.insert(index, item)

    def __call__(self, inputs):
        if len(self) == 0:
            raise RuntimeError("Can not run on empty transform")
        for t in self:
            inputs = t(inputs)
        return inputs


class SomeOf(Compose):

    def __init__(self, num, *transforms):
        transforms = list(itertools.combinations(transforms, num))
        super(SomeOf, self).__init__(*transforms)


class OneOf(SomeOf):

    def __init__(self, *transforms):
        super(OneOf, self).__init__(num=1, *transforms)


class Sometimes(Compose):

    def __call__(self, inputs):
        if random.randint(0, 1):
            return super(Sometimes, self).__call__(inputs)
        return inputs


class RandomOrder(Compose):
    def __call__(self, inputs):
        order = list(range(len(self.transforms)))
        random.shuffle(order)
        for i in order:
            inputs = self.transforms[i](inputs)
        return inputs
