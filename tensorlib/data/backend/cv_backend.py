import math
import random
import numpy as np
import cv2
import io
import base64
from PIL import Image


__all__ = ['imread', 'imwrite', 'keep_aspect_resize_padding',
           'keep_aspect_resize_points', 'resize_points', 'resize',
           'random_brightness', 'random_contrast', 'random_expand',
           'random_hue', 'random_saturation', 'ResizeMethod',
           'crop_resize', 'random_flip', 'crop_resize_points']


class ResizeMethod:
    BILINEAR = cv2.INTER_LINEAR
    NEAREST = cv2.INTER_NEAREST
    BICUBIC = cv2.INTER_CUBIC
    AREA = cv2.INTER_AREA


def _img_data_to_arr(img_data):
    f = io.BytesIO()
    f.write(img_data)
    img_arr = np.array(Image.open(f))
    return img_arr


def _img_b64_to_arr(img_b64):
    img_data = base64.b64decode(img_b64)
    img_arr = _img_data_to_arr(img_data)
    return img_arr


def imread(img_path):
    with open(img_path, 'rb') as f:
        img_data = f.read()
        img_data = base64.b64encode(img_data).decode('utf-8')
    img = _img_b64_to_arr(img_data)
    return img


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


def keep_aspect_resize_points(
        points,
        image,
        target_size,
        random_seed=None,
        is_limit=True):
    raw_h, raw_w = tuple(image.shape[:2])
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
    result_points = np.floor(np.concatenate([xs, ys], axis=-1)).astype(np.int64)
    result_points = np.concatenate([result_points, points[..., 2:]], axis=-1)
    return result_points


def crop_resize_points(
        points,
        image,
        crop_size):
    h, w = crop_size
    ih, iw, _ = image.shape
    ratio = max(h / ih, w / iw)
    size = (int(round(ih * ratio)), int(round(iw * ratio)))
    if size[0] > h:
        range_h = np.random.randint(0, size[0] - h)
        points = resize_points(points, (ih, iw), crop_size)
        _points = [[int(point[0]), int(point[1] - range_h), int(point[2])]
                   if range_h <= point[1] <= range_h + h - 1
                   else [0, 0, 0] for point in np.reshape(points, (-1, 3))]
        points = np.reshape(_points, points.shape)
    elif size[1] > w:
        range_w = np.random.randint(0, size[1] - w)
        points = resize_points(points, (ih, iw), crop_size)
        _points = [[int(point[0] - range_w), int(point[1]), int(point[2])]
                   if range_w <= point[0] <= range_w + w - 1
                   else [0, 0, 0] for point in np.reshape(points, (-1, 3))]
        points = np.reshape(_points, points.shape)
    return points


def resize_points(
        points,
        image_size,
        target_size):
    ih, iw = image_size
    h, w = target_size
    ratio_h = h / ih
    ratio_w = w / iw
    _points = [[int(point[0] * ratio_w + 0.5), int(point[1] * ratio_h + 0.5),
                int(point[2])] if 0 <= point[0] <= w - 1 and 0 <= point[1] <= h - 1
               else [0, 0, 0] for point in points.reshape(-1, 3)]
    points = np.floor(np.reshape(_points, points.shape))
    return np.asarray(points, dtype=np.int64)


def keep_aspect_resize_padding(
        image,
        resize_height,
        resize_width,
        random_seed=None,
        method=ResizeMethod.BILINEAR,
        border_value=(0, 0, 0)):
    raw_aspect = float(image.shape[1]) / image.shape[0]
    resize_aspect = float(resize_width) / resize_height
    if raw_aspect > resize_aspect:
        height = math.floor(resize_width / raw_aspect)
        resize_img = resize(image=image, size=(height, resize_width), method=method)
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
        resize_img = resize(image=image, size=(resize_height, width), method=method)
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


def crop_resize(
        image,
        crop_size,
        method=ResizeMethod.BILINEAR):
    h, w = crop_size
    ih, iw, _ = image.shape
    ratio = max(h / ih, w / iw)
    size = (int(round(ih * ratio)), int(round(iw * ratio)))
    if size[0] > h:
        range_h = np.random.randint(0, size[0] - h)
        image = resize(image, size, method=method)[range_h: range_h + h, :, :]
    elif size[1] > w:
        range_w = np.random.randint(0, size[1] - w)
        image = resize(image, size, method=method)[:, range_w: range_w + w, :]
    return image


def resize(
        image,
        size,
        method=ResizeMethod.BILINEAR):
    return cv2.resize(src=image, dsize=size[::-1], interpolation=method)


def random_expand(
        image,
        max_expand_ratio,
        mean_value,
        return_box=False):
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


def random_brightness(
        image,
        prob,
        delta):
    if random.random() > prob:
        delta = random.uniform(-delta, delta)
        delta = np.full_like(image, delta)
        image = cv2.add(image, delta)
    return image


def random_contrast(
        image,
        prob,
        lower,
        upper):
    if random.random() > prob:
        delta = random.uniform(lower, upper)
        image = cv2.addWeighted(image, delta, 0, 0, 0)
    return image


def random_saturation(
        image,
        prob,
        lower,
        upper):
    if random.random() > prob:
        delta = random.uniform(lower, upper)
        if math.fabs(delta - 1.) > 1e-3:
            hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hsv_img[:, :, 1] = cv2.addWeighted(hsv_img[:, :, 1], delta, 0, 0, 0)
            image = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    return image


def random_hue(
        image,
        prob,
        delta,
        channel_format="BGR"):
    if random.random() > prob:
        delta = random.uniform(-delta, delta)
        if math.fabs(delta) > 0.:
            hsv_img = cv2.cvtColor(
                image, cv2.COLOR_RGB2HSV
                if channel_format is 'RGB' else cv2.COLOR_BGR2HSV)
            hsv_img[:, :, 0] = cv2.add(hsv_img[:, :, 0], delta)
            image = cv2.cvtColor(
                hsv_img, cv2.COLOR_HSV2RGB
                if channel_format is 'RGB' else cv2.COLOR_HSV2BGR)
    return image


def random_flip(
        image,
        prob,
        flip_code):
    if random.random() > prob:
        image = cv2.flip(image, flip_code)
    return image


def random_affine(
        image,
        size,
        max_sum,
        max_angle,
        border_value):
    max_angle = np.array(max_angle, np.float32)
    angles = np.random.rand(3) * max_angle
    if np.sum(angles) > max_sum:
        angles = angles(angles / np.sum(angles)) * (max_angle / np.sum(max_angle))
    h, w = size
    ih, iw = image.shape[:2]
    coord = np.array([[0, iw, iw, 0], [0, 0, ih, ih]], dtype=np.float32)
    t_matrix = offset_transform(h, w, coord)


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
