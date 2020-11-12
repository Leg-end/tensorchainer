from tensorlib.contrib.data_flow.augmenter import Aug, check_number_param
import random
import numpy as np
import cv2
import math


class RandomBrightness(Aug):
    def __init__(self, prob, delta=32.):
        super(RandomBrightness, self).__init__()
        self.prob = check_number_param(prob, 'prob')
        self.delta = check_number_param(delta, 'delta')

    def augment(self, record):
        if record.image is None:
            raise ValueError("")
        prob = random.random()
        if prob < self.prob:
            delta = random.uniform(-self.delta, self.delta)
            delta_array = np.full_like(record.image, abs(delta))
            if delta > 0.:
                record.image = cv2.add(record.image, delta_array)
            else:
                record.image = cv2.subtract(record.image, delta_array)

        return record


class RandomHue(Aug):
    def __init__(self, hue_prob, hue_delta, color_channel="RGB"):
        super(RandomHue, self).__init__()
        self.hue_prob = check_number_param(hue_prob, 'hue_prob')
        self.hue_delta = check_number_param(hue_delta, 'hue_delta')
        assert isinstance(color_channel, str), (
            "Expected parameter '%s' with type str, "
            "got %s." % ("color_channel", type(color_channel)))
        self.color_channel = color_channel

    def augment(self, record):
        if record.image is None:
            raise ValueError("")
        prob = random.random()
        if prob < self.hue_prob:
            delta = random.uniform(-self.hue_delta, self.hue_delta)
            if math.fabs(delta) > 0.:
                hsv_img = cv2.cvtColor(
                    record.image,
                    cv2.COLOR_RGB2HSV if self.color_channel is 'RGB' else cv2.COLOR_BGR2HSV)
                hsv_img[:, :, 0] = cv2.add(hsv_img[:, :, 0], delta)
                record.image = cv2.cvtColor(
                    hsv_img,
                    cv2.COLOR_HSV2RGB if self.color_channel is 'RGB' else cv2.COLOR_HSV2BGR)

        return record


class RandomContrast(Aug):
    def __init__(self,
                 prob,
                 lower=0.6,
                 upper=1.2):
        super(RandomContrast, self).__init__()
        self.prob = check_number_param(prob, "prob")
        self.lower = check_number_param(lower, "lower")
        self.upper = check_number_param(upper, "upper")

    def augment(self, record):
        if record.image is None:
            raise ValueError("")
        assert record.image.dtype.name == "uint8", (
            "Expected uint8 image from `record.image`, "
            "got dtype %s." % (record.image.dtype.name,))
        prob = random.random()
        if prob < self.prob:
            delta = random.uniform(self.lower, self.upper)
            record.image = cv2.addWeighted(record.image, delta, 0, 0, 0)
        return record

# class RandomContrast(Aug):
#     def __init__(self,
#                  prob,
#                  gamma=(0.7, 1.7)):
#         super(RandomContrast, self).__init__()
#         self.gamma = check_number_param(gamma, "gamma", list_or_tuple=True)
#         self.prob = prob
#
#     def augment(self, record):
#         if record.image is None:
#             raise ValueError("")
#         assert record.image.dtype.name == "uint8", (
#                 "Expected uint8 image from `record.image`, "
#                 "got dtype %s." % (record.image.dtype.name,))
#
#         prob = random.random()
#
#         if prob < self.prob:
#             record.image = _adjust_contrast(record.image, self.gamma)
#         return record
#
#
# def _adjust_contrast(image, gamma):
#     info = np.iinfo(image.dtype)
#     min_value = info.min
#     max_value = info.max
#
#     dynamic_range = max_value - min_value
#     value_range = np.linspace(0, 1.0, num=dynamic_range + 1,
#                               dtype=np.float32)
#     print(value_range)
#     table = ((value_range ** np.float32(gamma))
#              * dynamic_range)
#
#     table = np.clip(table, min_value, max_value).astype(image.dtype)
#     image = _apply_lut(image, table)
#     return image
#
#
# def _apply_lut(image, table):
#     shape = image.shape
#     channels = 1 if len(shape) == 2 else shape[-1]
#     if isinstance(table, list):
#         assert len(table) == channels, (
#             "Expected to get %d tables (one per channel), got %d instead." % (
#              channels, len(table)))
#         table = np.stack(table, axis=-1)
#
#     if table.shape == (256, channels):
#         table = table[np.newaxis, :, :]
#
#     assert table.shape == (256,) or table.shape == (1, 256, channels), (
#         "Expected 'table' to be any of the following: "
#         "A list of C (256,) arrays, an array of shape (256,), an array of "
#         "shape (256, C), an array of shape (1, 256, C). Transformed 'table' "
#         "up to shape %s for image with shape %s (C=%d)." % (
#                 table.shape, shape, channels))
#
#     assert image.dtype.name == "uint8", (
#             "Expected uint8 image, got dtype %s." % (image.dtype.name,))
#     assert table.dtype.name == "uint8", (
#             "Expected uint8 table, got dtype %s." % (table.dtype.name,))
#
#     image = cv2.LUT(image, table, dst=image)
#     return image
