import numpy as np
from PIL import Image
from tensorlib.contrib.visual_tool import color_utils
from tensorlib.contrib.visual_tool import draw_utils


def label_colormap(n_label=256, value=None):
    """Label colormap.

    Parameters
    ----------
    n_labels: int
        Number of labels (default: 256).
    value: float or int
        Value scale or value of label color in HSV space.

    Returns
    -------
    cmap: numpy.ndarray, (N, 3), numpy.uint8
        Label id to colormap.

    """

    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    cmap = np.zeros((n_label, 3), dtype=np.uint8)
    for i in range(0, n_label):
        id = i
        r, g, b = 0, 0, 0
        for j in range(0, 8):
            r = np.bitwise_or(r, (bitget(id, 0) << 7 - j))
            g = np.bitwise_or(g, (bitget(id, 1) << 7 - j))
            b = np.bitwise_or(b, (bitget(id, 2) << 7 - j))
            id = (id >> 3)
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b

    if value is not None:
        hsv = color_utils.rgb2hsv(cmap.reshape(1, -1, 3))
        if isinstance(value, float):
            hsv[:, 1:, 2] = hsv[:, 1:, 2].astype(float) * value
        else:
            assert isinstance(value, int)
            hsv[:, 1:, 2] = value
        cmap = color_utils.hsv2rgb(hsv).reshape(-1, 3)
    return cmap


def label2rgb(
    label,
    img=None,
    alpha=0.5,
    label_names=None,
    font_size=30,
    thresh_suppress=0,
    colormap=None,
    loc='centroid'):
    """Convert label to rgb.

    Parameters
    ----------
    label: numpy.ndarray, (H, W), int
        Label image.
    img: numpy.ndarray, (H, W, 3), numpy.uint8
        RGB image.
    alpha: float
        Alpha of RGB (default: 0.5).
    label_names: list of string
        Label id to label name.
    font_size: int
        Font size (default: 30).
    thresh_suppress: float
        Threshold of label ratio in the label image.
    colormap: numpy.ndarray, (M, 3), numpy.uint8
        Label id to color.
        By default, :func:`~imgviz.label_colormap` is used.
    loc: string
        Location of legend (default: 'centroid').
        'lt' and 'rb' are supported.

    Returns
    -------
    res: numpy.ndarray, (H, W, 3), numpy.uint8
        Visualized image.

    """
    if colormap is None:
        colormap = label_colormap()

    res = colormap[label]

    np.random.seed(1234)

    mask_unlabeled = label < 0
    res[mask_unlabeled] = \
        np.random.random(size=(mask_unlabeled.sum(), 3)) * 255

    if img is not None:
        if img.ndim == 2:
            img = color_utils.gray2rgb(img)
        res = (1 - alpha) * img.astype(float) + alpha * res.astype(float)
        res = np.clip(res.round(), 0, 255).astype(np.uint8)

    if label_names is None:
        return res

    if loc == 'centroid':
        for l in np.unique(label):
            if l == -1:
                continue  # unlabeled

            mask = label == l
            if 1. * mask.sum() / mask.size < thresh_suppress:
                continue
            y, x = np.array(_center_of_mass(mask), dtype=int)

            if label[y, x] != l:
                Y, X = np.where(mask)
                point_index = np.random.randint(0, len(Y))
                y, x = Y[point_index], X[point_index]

            text = label_names[l]
            height, width = draw_utils.text_size(text, size=font_size)
            color = color_utils.get_fg_color(res[y, x])
            res = draw_utils.text(
                res,
                yx=(y - height // 2, x - width // 2),
                text=text,
                color=color,
                size=font_size,
            )
    elif loc in ['rb', 'lt']:
        unique_labels = np.unique(label)
        unique_labels = unique_labels[unique_labels != -1]
        text_sizes = np.array([
            draw_utils.text_size(label_names[l], font_size)
            for l in unique_labels
        ])
        text_height, text_width = text_sizes.max(axis=0)
        legend_height = text_height * len(unique_labels) + 5
        legend_width = text_width + 40

        height, width = label.shape[:2]
        legend = np.zeros((height, width, 3), dtype=np.uint8)
        if loc == 'rb':
            aabb2 = np.array([height - 5, width - 5], dtype=float)
            aabb1 = aabb2 - (legend_height, legend_width)
        elif loc == 'lt':
            aabb1 = np.array([5, 5], dtype=float)
            aabb2 = aabb1 + (legend_height, legend_width)
        else:
            raise ValueError('unexpected loc: {}'.format(loc))
        legend = draw_utils.rectangle(
            legend, aabb1, aabb2, fill=(255, 255, 255))

        alpha = 0.5
        y1, x1 = aabb1.round().astype(int)
        y2, x2 = aabb2.round().astype(int)
        res[y1:y2, x1:x2] = \
            alpha * res[y1:y2, x1:x2] + alpha * legend[y1:y2, x1:x2]

        for i, l in enumerate(unique_labels):
            box_aabb1 = aabb1 + (i * text_height + 5, 5)
            box_aabb2 = box_aabb1 + (text_height - 10, 20)
            res = draw_utils.rectangle(
                res,
                aabb1=box_aabb1,
                aabb2=box_aabb2,
                fill=colormap[l]
            )
            res = draw_utils.text(
                res,
                yx=aabb1 + (i * text_height, 30),
                text=label_names[l],
                size=font_size,
            )
    else:
        raise ValueError('unsupported loc: {}'.format(loc))

    return res


def _center_of_mass(mask):
    assert mask.ndim == 2 and mask.dtype == bool
    mask = 1. * mask / mask.sum()
    dx = np.sum(mask, 0)
    dy = np.sum(mask, 1)
    cx = np.sum(dx * np.arange(mask.shape[1]))
    cy = np.sum(dy * np.arange(mask.shape[0]))
    return cy, cx


def segment2rgb(segment_map, image, label_names=None):
    """
    Parameters
    ----------
    segment_map:
            (H, W) ndarray which dtype is `uint8`.
    image:
            (H,W,3) ndarray
            image onto which to draw the segmentation map. Expected dtype
            is `uint8`.
    label_names:
    Returns
    -------
        A rendered overlays RGB image which is ndarray.dtype as `uint8`.
    """

    assert segment_map.ndim == 2, (
            "Expected to be draw with 2-dimensional segment_map, got with %d "
            "dimensions." % (segment_map.ndim,))

    assert segment_map.dtype.name == 'uint8', (
            "Expected to get segment_map with dtype uint8, got dtype %s." % (
                segment_map.dtype.name,))

    assert image.ndim == 3, (
            "Expected to draw on 3-dimensional image, got image with %d "
            "dimensions." % (image.ndim,))

    assert image.shape[2] == 3, (
            "Expected to draw on RGB image, got image with %d channels "
            "instead." % (image.shape[2],))

    assert image.dtype.name == 'uint8', (
        "Expected to get RGB image with dtype uint8, got dtype %s." % (
                image.dtype.name,))

    return label2rgb(label=segment_map, img=color_utils.rgb2gray(image), label_names=label_names, loc='rb')


def vis_save(path, vis_img):
    """
    Parameters
    ----------
    path: A filename (string), path.Path object or file object.
    vis_img: A numpy.ndarray, shape of (H, W, 3), dtype is numpy.uint8
    """
    Image.fromarray(vis_img).save(path)
