import tensorflow as tf


__all__ = ['keep_aspect_resize_padding', 'resize',
           'random_brightness', 'random_contrast', 'random_expand',
           'random_hue', 'random_saturation', 'ResizeMethod',
           'crop_resize', 'random_flip', 'random_scale', 'image_padding']


class ResizeMethod:
    BILINEAR = tf.image.ResizeMethod.BILINEAR
    NEAREST = tf.image.ResizeMethod.NEAREST_NEIGHBOR
    BICUBIC = tf.image.ResizeMethod.BICUBIC
    AREA = tf.image.ResizeMethod.AREA


def keep_aspect_resize_padding(
        image,
        resize_height,
        resize_width,
        method=ResizeMethod.BILINEAR):
    return tf.image.resize_image_with_pad(
        image=image, target_height=resize_height,
        target_width=resize_width, method=method)


def resize(
        image,
        size,
        method=ResizeMethod.BILINEAR):
    return tf.image.resize_images(
        images=image, size=size, method=method)


def crop_resize(
        image,
        crop_size,
        method=ResizeMethod.BILINEAR):
    return tf.image.crop_and_resize(
        image, crop_size=crop_size, method=method)


def image_padding(
        image,
        offset_height,
        offset_width,
        target_height,
        target_width,
        pad_value):
    # throw an exception when padding
    with tf.name_scope('image_padding'):
        image = tf.convert_to_tensor(image, name='images')
        image_dtype = image.dtype
        pad_value = tf.convert_to_tensor(pad_value, dtype=image_dtype)

        image_rank = tf.rank(image)
        height, width = tf.shape(image)[0], tf.shape(image)[1]
        rank_assert = tf.Assert(tf.logical_or(tf.equal(image_rank, 3), tf.equal(image_rank, 4)),
                                ['Wrong images tensor rank. should be 3 or 4 !'])
        with tf.control_dependencies([rank_assert]):
            image -= pad_value
        h_range_assert = tf.Assert(tf.greater_equal(target_height, height),
                                   ['`target_height` must be >= height of `images`'])
        w_range_assert = tf.Assert(tf.greater_equal(target_width, width),
                                   ['`target_width` must be >= width of `images`'])
        with tf.control_dependencies([h_range_assert]):
            top_padding = target_height - offset_height - height
        with tf.control_dependencies([w_range_assert]):
            right_padding = target_width - offset_width - width

        offset_assert = tf.Assert(tf.logical_and(
            tf.greater_equal(top_padding, 0), tf.greater_equal(right_padding, 0)),
            ['Target size not possible with the given target offsets'])
        height_param = tf.stack([offset_height, top_padding])
        width_param = tf.stack([offset_width, right_padding])
        channel_param = tf.constant([0, 0])

        with tf.control_dependencies([offset_assert]):
            paddings = tf.stack([height_param, width_param, channel_param])

        padded_image = tf.pad(image, paddings=paddings)
        outputs = padded_image + pad_value
        return outputs


def random_scale(
        image,
        min_scale_factor=1.0,
        max_scale_factor=1.0,
        step=0.,
        method=ResizeMethod.NEAREST,
        seed=None):

    if min_scale_factor < 0. or min_scale_factor > max_scale_factor:
        raise ValueError("Unexpected value of `min_scale_factor`")

    with tf.name_scope('image_scale'):
        image = tf.convert_to_tensor(image, name='images')
        image_rank = tf.rank(image)
        image_shape = tf.shape(image)

        rank_assert = tf.Assert(tf.logical_or(tf.equal(image_rank, 3), tf.equal(image_rank, 4)),
                                ['Wrong images tensor rank. should be 3 or 4 !'])

        _get_scale_shape = lambda scale: tf.cast(
                tf.cast([image_shape[0], image_shape[1]], dtype=tf.float32) * scale,
                dtype=tf.int32)

        if min_scale_factor == max_scale_factor:
            scaled_shape = _get_scale_shape(min_scale_factor)
        else:
            if step == 0.:
                scale_factor = tf.random_uniform([1],
                                                 minval=min_scale_factor,
                                                 maxval=max_scale_factor,
                                                 seed=seed)
                scaled_shape = _get_scale_shape(scale_factor)
            else:
                num_steps = int((max_scale_factor - min_scale_factor) / step + 1)
                scale_factors = tf.lin_space(min_scale_factor, max_scale_factor, num_steps)
                scale_factor = tf.random_shuffle(scale_factors, seed=seed)
                scaled_shape = _get_scale_shape(scale_factor)
        with tf.control_dependencies([rank_assert]):
            image = tf.image.resize(image, size=scaled_shape, method=method, align_corners=True)
    return image


def random_expand(
        image,
        max_expand_ratio,
        mean_value,
        return_box=False,
        seed=None):

    height = tf.to_float(tf.shape(image)[0])
    width = tf.to_float(tf.shape(image)[1])
    expand_ratio = tf.random_uniform(shape=(), minval=1., maxval=max_expand_ratio, seed=seed)
    expand_height = height * expand_ratio
    expand_width = width * expand_ratio

    h_offset = tf.floor(tf.random_uniform(shape=(), maxval=expand_height - height, seed=seed))
    w_offset = tf.floor(tf.random_uniform(shape=(), maxval=expand_width - width, seed=seed))

    expand_image = image_padding(image=image,
                                 offset_height=tf.to_int32(h_offset),
                                 offset_width=tf.to_int32(w_offset),
                                 target_height=tf.to_int32(expand_height),
                                 target_width=tf.to_int32(expand_width), pad_value=mean_value)
    if return_box:
        x_l = -w_offset / width
        y_t = -h_offset / height
        x_r = (expand_width - w_offset) / width
        y_b = (expand_height - h_offset) / height
        expand_box = tf.stack([x_l, y_t, x_r, y_b])
        return expand_image, expand_box
    else:
        return expand_image


def random_hue(
        image,
        hue_prob,
        hue_delta,
        seed=None):
    with tf.name_scope("image_hue"):
        prob = _tf_random(seed=seed)
        image = tf.cond(tf.less(prob, hue_prob),
                        true_fn=lambda: tf.image.random_hue(image=image, max_delta=hue_delta),
                        false_fn=lambda: image)
    return image


def random_saturation(
        image,
        prob,
        lower,
        upper,
        seed=None):
    with tf.name_scope('image_saturation'):
        p = _tf_random(seed=seed)
        image = tf.cond(tf.less(p, prob),
                        true_fn=lambda: tf.image.random_saturation(
                            image=image, lower=lower, upper=upper),
                        false_fn=lambda: image)
    return image


def random_contrast(
        image,
        prob,
        lower,
        upper,
        seed=None):
    with tf.name_scope('image_contrast'):
        p = _tf_random(seed=seed)
        image = tf.cond(tf.less(p, prob),
                        true_fn=lambda: tf.image.random_contrast(
                            image=image, lower=lower, upper=upper),
                        false_fn=lambda: image)
    return image


def random_brightness(
        image,
        prob,
        delta,
        seed=None):
    with tf.name_scope('image_brightness'):
        p = _tf_random(seed=seed)
        image = tf.cond(tf.less(p, prob),
                        true_fn=lambda: tf.image.random_brightness(
                            image=image, max_delta=delta / 255),
                        false_fn=lambda: image)
    return image


def random_flip(
        image,
        prob,
        flip_code,
        seed=None):
    with tf.name_scope('image_flip'):
        p = _tf_random(seed=seed)
        image = tf.cond(tf.greater(p, prob),
                        true_fn=lambda: tf.image.random_flip_left_right(
                            image=image) if flip_code
                        else tf.image.random_flip_up_down(
                            image=image),
                        false_fn=lambda: image)
    return image


def _tf_random(seed=None):
    return tf.random_uniform(shape=(), seed=seed)
