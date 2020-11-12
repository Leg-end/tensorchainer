import random
import math
import numpy as np
import cv2
import base64
import io
from PIL import Image


class ResizeMethod:
    """
    This class cannot be inherited, if so, an error will be reported
    """
    BILINEAR = cv2.INTER_LINEAR
    NEAREST = cv2.INTER_NEAREST
    BICUBIC = cv2.INTER_CUBIC
    AREA = cv2.INTER_AREA

    def __new__(cls, *args, **kwargs):
        if cls != ResizeMethod:
            raise Exception("You should not inherit from ResizeMethod; Got subclass(es): "
                            + str(ResizeMethod.__subclasses__()))
        return super(ResizeMethod, cls).__new__(cls, *args, **kwargs)

    def __init__(self, x):
        self.x = x


def keep_aspect_resize_padding(image, resize_height, resize_width, random_seed=None,
                               interp=ResizeMethod.BILINEAR, border_value=(0, 0, 0)):
    raw_aspect = float(image.shape[1]) / image.shape[0]
    resize_aspect = float(resize_width) / resize_height
    if raw_aspect > resize_aspect:
        height = math.floor(resize_width / raw_aspect)
        resize_img = resize(image=image, size=(height, resize_width), interp=interp)
        h = resize_img.shape[0]
        if random_seed:
            random.seed(random_seed)
            padding = random.randint(0, resize_height-h)
        else:
            padding = math.floor((resize_height - h) / 2.0)
        resize_img = cv2.copyMakeBorder(src=resize_img, top=padding, bottom=resize_height - h - padding, left=0,
                                        right=0, borderType=cv2.BORDER_CONSTANT, value=border_value)
    else:
        width = math.floor(raw_aspect * resize_height)
        resize_img = resize(image=image, size=(resize_height, width), interp=interp)
        w = resize_img.shape[1]
        if random_seed:
            random.seed(random_seed)
            padding = random.randint(0, resize_width-w)
        else:
            padding = math.floor((resize_width - w) / 2.0)
        resize_img = cv2.copyMakeBorder(src=resize_img, top=0, bottom=0, left=padding,
                                        right=resize_width - w - padding,
                                        borderType=cv2.BORDER_CONSTANT, value=border_value)

    return resize_img


def resize(image, size, interp=ResizeMethod.BILINEAR):
    return cv2.resize(src=image, dsize=size[::-1], interpolation=interp)


def random_expand(image, max_expand_ratio, mean_value, return_box=False):
    expand_ratio = random.uniform(1., max_expand_ratio)
    height, width, _ = image.shape
    expand_height = int(height * expand_ratio)
    expand_width = int(width * expand_ratio)

    h_offset = int(math.floor(random.uniform(0., expand_height - height)))
    w_offset = int(math.floor(random.uniform(0., expand_width - width)))

    fill_value = mean_value if mean_value is not None else [0] * 3
    expand_image = np.full(shape=(expand_height, expand_width, 3), fill_value=fill_value, dtype=image.dtype)
    expand_image[h_offset: h_offset + height, w_offset: w_offset + width, :] = image

    if return_box:
        x_l = -w_offset / float(width)
        y_t = -h_offset / float(height)
        x_r = (expand_width - w_offset) / float(width)
        y_b = (expand_height - h_offset) / float(height)
        expand_box = [x_l, y_t, x_r, y_b]
        return expand_image, expand_box
    else:
        return expand_image


def random_brightness(image, brightness_prob, brightness_delta):

    def _random_brightness(img, delta):
        delta = random.uniform(-delta, delta)
        delta_array = np.full_like(img, abs(delta))
        if delta > 0.:
            img = cv2.add(img, delta_array)
        else:
            img = cv2.subtract(img, delta_array)
        return img

    prob = random.random()
    if prob < brightness_prob:
        image = _random_brightness(img=image, delta=brightness_delta)
        return image
    return image.astype(np.float32)


def random_contrast(image, contrast_prob, lower, upper):

    def _random_contrast(img, _lower, _upper):
        delta = random.uniform(_lower, _upper)
        img = cv2.addWeighted(img, delta, 0, 0, 0)
        return img

    prob = random.random()
    if prob < contrast_prob:
        image = _random_contrast(img=image, _lower=lower, _upper=upper)
        return image
    return image.astype(np.float32)


def random_saturation(image, saturation_prob, lower, upper):

    def _random_saturation(img, _lower, _upper):
        delta = random.uniform(_lower, _upper)
        if math.fabs(delta - 1.) > 1e-3:
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hsv_img[:, :, 1] = cv2.addWeighted(hsv_img[:, :, 1], delta, 0, 0, 0)
            img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
            return img
        return img

    prob = random.random()
    if prob < saturation_prob:
        image = _random_saturation(img=image, _lower=lower, _upper=upper)
        return image
    return image


def random_hue(image, hue_prob, hue_delta):

    def _random_hue(img, delta):
        delta = random.uniform(-delta, delta)
        if math.fabs(delta) > 0.:
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hsv_img[:, :, 0] = cv2.add(hsv_img[:, :, 0], delta)
            img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
            return img
        return img

    prob = random.random()
    if prob < hue_prob:
        image = _random_hue(img=image, delta=hue_delta)
        return image
    return image


def imread(img_path):
    with open(img_path, 'rb') as f:
        img_data = f.read()
        img_data = base64.b64encode(img_data).decode('utf-8')
    img = _img_b64_to_arr(img_data)
    return img


def _img_data_to_arr(img_data):
    f = io.BytesIO()
    f.write(img_data)
    img_arr = np.array(Image.open(f))
    return img_arr


def _img_b64_to_arr(img_b64):
    img_data = base64.b64decode(img_b64)
    img_arr = _img_data_to_arr(img_data)
    return img_arr


def imwrite(filename, image, mode='P'):
    import imgviz
    if image.min() >= -1 and image.max() <= 255:
        image = Image.fromarray(image.astype(np.uint8), mode=mode)
        colomap = imgviz.label_colormap()
        image.putpalette(colomap.flatten())
        image.save(filename)
    else:
        raise ValueError(
            '[%s] Cannot save the pixel-wise class label as PNG. ' % filename)
