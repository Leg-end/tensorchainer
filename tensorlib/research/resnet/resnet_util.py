import tensorlib as lib


def conv2d_same(depth,
                kernel_size,
                stride,
                rate=1,
                activation=None,
                name='conv2d_same'):
    if stride == 1:
        return lib.contrib.Conv2D(depth, kernel_size=kernel_size,
                                  strides=stride, dilations=rate,
                                  activation=activation, name=name)
    else:
        return lib.Sequential(lib.layers.Pad2D(kernel_size=kernel_size, rate=rate),
                              lib.contrib.Conv2D(depth, kernel_size=kernel_size,
                                                 strides=stride, dilations=rate,
                                                 padding='VALID', activation=activation,
                                                 name=name))


def root_block(depth_multiplier=1.0):
    return lib.Sequential(
        conv2d_same(
            depth=int(64 * depth_multiplier),
            kernel_size=3,
            stride=2,
            name='conv1_1'),
        conv2d_same(
            depth=int(64 * depth_multiplier),
            kernel_size=3,
            stride=1,
            name='conv1_2'),
        conv2d_same(
            depth=int(128 * depth_multiplier),
            kernel_size=3,
            stride=1,
            name='conv1_3'))


def resnet_arg_scope(
        weight_decay=0.0001,
        batch_norm_decay=0.997,
        batch_norm_epsilon=1e-5,
        activation='relu',
        use_batch_norm=True):
    batch_norm_params = {
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
    }

    with lib.engine.arg_scope(
            [lib.contrib.Conv2D],
            kernel_regularizer=lib.regularizers.l2(weight_decay),
            kernel_initializer='truncated_normal',
            activation=activation,
            normalizer_fn=lib.layers.BatchNorm if use_batch_norm else None):
        with lib.engine.arg_scope([lib.layers.BatchNorm], **batch_norm_params):
            with lib.engine.arg_scope([lib.layers.MaxPool2D], padding='SAME') as arg_sc:
                return arg_sc
