import tensorlib as lib
import tensorflow as tf
from tensorlib.research.resnet_v1_beta import resnet_v1_utils
import functools

_DEFAULT_MULTI_GRID = [1, 1, 1]
_DEFAULT_MULTI_GRID_RESNET_18 = [1, 1]


@lib.engine.add_arg_scope
def bottleneck(inputs,
               depth,
               depth_bottleneck,
               stride,
               unit_rate=1,
               rate=1,
               scope=None):
    with lib.graph_scope(scope, 'bottleneck_v1', [inputs]) as handler:
        inputs = handler.inputs
        depth_in = lib.engine.int_shape(inputs)[-1]
        if depth == depth_in:
            shortcut = resnet_v1_utils.sub_sample(
                inputs, stride, 'shortcut')
        else:
            shortcut = resnet_v1_utils.ws_conv2d(
                inputs,
                out_channels=depth,
                kernel_size=1,
                stride=stride,
                activation=None,
                name='shortcut')
        residual = resnet_v1_utils.ws_conv2d(
            inputs,
            out_channels=depth_bottleneck,
            kernel_size=1,
            stride=1,
            name='conv1')
        residual = resnet_v1_utils.conv2d_same(
            residual,
            out_channels=depth_bottleneck,
            kernel_size=3,
            stride=stride,
            rate=rate * unit_rate,
            name='conv2')
        residual = resnet_v1_utils.ws_conv2d(
            residual,
            out_channels=depth,
            kernel_size=1,
            stride=1,
            activation=None,
            name='conv3')
        outputs = lib.layers.ElementWise('add', activation='relu')(shortcut, residual)
        handler.outputs = outputs
    return outputs


@lib.engine.add_arg_scope
def lite_bottleneck(inputs,
                    depth,
                    stride,
                    unit_rate=1,
                    rate=1,
                    scope=None):
    with lib.graph_scope(scope, 'lite_bottleneck_v1', [inputs]) as handler:
        inputs = handler.inputs
        depth_in = lib.engine.int_shape(inputs)[-1]
        if depth == depth_in:
            shortcut = resnet_v1_utils.sub_sample(
                inputs, stride, 'shortcut')
        else:
            shortcut = resnet_v1_utils.ws_conv2d(
                inputs,
                out_channels=depth,
                kernel_size=1,
                stride=stride,
                activation=None,
                name='shortcut')
        residual = resnet_v1_utils.conv2d_same(
            inputs,
            out_channels=depth,
            kernel_size=3,
            stride=1,
            rate=rate * unit_rate,
            name='conv1')
        with lib.engine.arg_scope([resnet_v1_utils.ws_conv2d], activation=None):
            residual = resnet_v1_utils.conv2d_same(
                residual,
                out_channels=depth,
                kernel_size=3,
                stride=stride,
                rate=rate * unit_rate,
                name='conv2')
        outputs = lib.layers.ElementWise('add', activation='relu')(shortcut, residual)
        handler.outputs = outputs
    return outputs


def ResNet_V1_beta_block(scope, base_depth, num_units, stride):
    return resnet_v1_utils.Block(scope, bottleneck, [{
        'depth': base_depth * 4,
        'depth_bottleneck': base_depth,
        'stride': 1,
        'unit_rate': 1
    }] * (num_units - 1) + [{
        'depth': base_depth * 4,
        'depth_bottleneck': base_depth,
        'stride': stride,
        'unit_rate': 1
    }])


def ResNet_V1_small_beta_block(scope, base_depth, num_units, stride):
    block_args = []
    for _ in range(num_units - 1):
        block_args.append({'depth': base_depth, 'stride': 1, 'unit_rate': 1})
    block_args.append({'depth': base_depth, 'stride': stride, 'unit_rate': 1})
    return resnet_v1_utils.Block(scope, lite_bottleneck, block_args)


def ResNet_V1_beta(blocks,
                   input_shape=(224, 224, 3),
                   num_classes=None,
                   is_training=None,
                   base_only=False,
                   global_pool=True,
                   extract_blocks=None,
                   output_stride=None,
                   root_block_fn=None,
                   name=None):
    if output_stride is not None and output_stride % 4 != 0:
        raise ValueError("The output stride needs to be a multiple of 4.")
    if extract_blocks is not None:
        extract_blocks = lib.utils.to_list(extract_blocks)
    else:
        extract_blocks = []
    if not base_only and output_stride not in [32, None]:
        raise ValueError("As the `base_only` is set to `False`, `output_stride` can only be 32 or None, "
                         "but given %d." % output_stride)
    with lib.hooks.ExtractHook(in_names='input', out_names=extract_blocks) as hook:
        with tf.name_scope(name, 'resnet_v1') as scope:
            inputs = lib.engine.Input(input_shape=input_shape)
            if is_training is not None:
                arg_scope = lib.engine.arg_scope(
                    [lib.layers.BatchNorm], trainable=is_training)
            else:
                arg_scope = lib.engine.arg_scope([])
            with arg_scope:
                net = inputs
                if root_block_fn is None:
                    net = resnet_v1_utils.conv2d_same(
                        net,
                        out_channels=64,
                        kernel_size=7,
                        stride=2,
                        name='conv1')
                else:
                    net = root_block_fn(net)
                net = lib.layers.MaxPool2D(
                    kernel_size=3,
                    strides=2,
                    padding='SAME',
                    name='pool1')(net)
                net = resnet_v1_utils.stack_blocks_dense(
                    net, blocks, output_stride=output_stride)
            if not base_only:
                if global_pool:
                    net = lib.layers.GlobalAvgPool(name='pool5')(net)
                if num_classes is not None:
                    net = resnet_v1_utils.ws_conv2d(
                        net,
                        out_channels=num_classes,
                        kernel_size=1,
                        activation=None,
                        normalizer_fn=None,
                        use_weight_standardization=False,
                        name='logits')
                    net = lib.layers.Lambda(
                        lambda x: tf.squeeze(x, axis=[1, 2]),
                        name='spatial_squeeze')(net)
    if extract_blocks:
        inputs, outputs = hook.get_extract()
        return lib.engine.Network(inputs=inputs, outputs=outputs, trainable=is_training, name=scope)
    else:
        return lib.engine.Network(inputs=inputs, outputs=net, trainable=is_training, name=scope)


def ResNet_V1_18(input_shape=(224, 224, 3),
                 num_classes=None,
                 is_training=None,
                 global_pool=False,
                 base_only=False,
                 min_base_depth=8,
                 depth_multiplier=1,
                 output_stride=None,
                 extract_blocks=None,
                 multi_grid=None,
                 name='resnet_v1_18'):
    if multi_grid is None:
        multi_grid = _DEFAULT_MULTI_GRID_RESNET_18
    elif len(multi_grid) != 2:
        raise ValueError("Expect multi_grid to have length 2.")
    depth_func = lambda d: max(int(d * depth_multiplier), min_base_depth)
    block4_args = []
    for rate in multi_grid:
        block4_args.append({'depth': depth_func(512), 'stride': 1, 'unit_rate': rate})

    blocks = [
        ResNet_V1_small_beta_block(
            'block1', base_depth=depth_func(64), num_units=2, stride=2),
        ResNet_V1_small_beta_block(
            'block2', base_depth=depth_func(128), num_units=2, stride=2),
        ResNet_V1_small_beta_block(
            'block3', base_depth=depth_func(256), num_units=2, stride=2),
        resnet_v1_utils.Block('block4', lite_bottleneck, block4_args)]
    return ResNet_V1_beta(input_shape=input_shape,
                          blocks=blocks,
                          num_classes=num_classes,
                          is_training=is_training,
                          global_pool=global_pool,
                          base_only=base_only,
                          output_stride=output_stride,
                          extract_blocks=extract_blocks,
                          name=name)


def ResNet_V1_18_beta(input_shape=(224, 224, 3),
                      num_classes=None,
                      is_training=None,
                      global_pool=False,
                      base_only=False,
                      min_base_depth=8,
                      depth_multiplier=1,
                      root_depth_multiplier=0.25,
                      output_stride=None,
                      extract_blocks=None,
                      multi_grid=None,
                      name='resnet_v1_18'):
    if multi_grid is None:
        multi_grid = _DEFAULT_MULTI_GRID_RESNET_18
    elif len(multi_grid) != 2:
        raise ValueError("Expect multi_grid to have length 2.")
    depth_func = lambda d: max(int(d * depth_multiplier), min_base_depth)
    block4_args = []
    for rate in multi_grid:
        block4_args.append({'depth': depth_func(512), 'stride': 1, 'unit_rate': rate})

    blocks = [
        ResNet_V1_small_beta_block(
            'block1', base_depth=depth_func(64), num_units=2, stride=2),
        ResNet_V1_small_beta_block(
            'block2', base_depth=depth_func(128), num_units=2, stride=2),
        ResNet_V1_small_beta_block(
            'block3', base_depth=depth_func(256), num_units=2, stride=2),
        resnet_v1_utils.Block('block4', lite_bottleneck, block4_args)]
    return ResNet_V1_beta(input_shape=input_shape,
                          blocks=blocks,
                          num_classes=num_classes,
                          is_training=is_training,
                          global_pool=global_pool,
                          base_only=base_only,
                          root_block_fn=functools.partial(
                              resnet_v1_utils.root_block_fn,
                              depth_multiplier=root_depth_multiplier),
                          output_stride=output_stride,
                          extract_blocks=extract_blocks,
                          name=name)


def ResNet_V1_50(input_shape=(224, 224, 3),
                 num_classes=None,
                 is_training=None,
                 base_only=False,
                 global_pool=False,
                 min_base_depth=8,
                 depth_multiplier=1,
                 output_stride=None,
                 extract_blocks=None,
                 multi_grid=None,
                 name='resnet_v1_50'):
    if multi_grid is None:
        multi_grid = _DEFAULT_MULTI_GRID
    elif len(multi_grid) != 3:
        raise ValueError("Expect multi_grid to have length 3.")
    depth_func = lambda d: max(int(d * depth_multiplier), min_base_depth)
    blocks = [
        ResNet_V1_beta_block(
            'block1', base_depth=depth_func(64), num_units=3, stride=2),
        ResNet_V1_beta_block(
            'block2', base_depth=depth_func(128), num_units=4, stride=2),
        ResNet_V1_beta_block(
            'block3', base_depth=depth_func(256), num_units=6, stride=2),
        resnet_v1_utils.Block('block4', bottleneck, [
            {'depth': 2048, 'depth_bottleneck': depth_func(512), 'stride': 1,
             'unit_rate': rate} for rate in multi_grid])]
    return ResNet_V1_beta(input_shape=input_shape,
                          blocks=blocks,
                          num_classes=num_classes,
                          is_training=is_training,
                          base_only=base_only,
                          global_pool=global_pool,
                          output_stride=output_stride,
                          extract_blocks=extract_blocks,
                          name=name)


def ResNet_V1_50_beta(input_shape=(224, 224, 3),
                      num_classes=None,
                      is_training=None,
                      base_only=False,
                      global_pool=False,
                      min_base_depth=8,
                      depth_multiplier=1,
                      output_stride=None,
                      extract_blocks=None,
                      multi_grid=None,
                      name='resnet_v1_50'):
    if multi_grid is None:
        multi_grid = _DEFAULT_MULTI_GRID
    elif len(multi_grid) != 3:
        raise ValueError("Expect multi_grid to have length 3.")
    depth_func = lambda d: max(int(d * depth_multiplier), min_base_depth)
    blocks = [
        ResNet_V1_beta_block(
            'block1', base_depth=depth_func(64), num_units=3, stride=2),
        ResNet_V1_beta_block(
            'block2', base_depth=depth_func(128), num_units=4, stride=2),
        ResNet_V1_beta_block(
            'block3', base_depth=depth_func(256), num_units=6, stride=2),
        resnet_v1_utils.Block('block4', bottleneck, [
            {'depth': 2048, 'depth_bottleneck': depth_func(512), 'stride': 1,
             'unit_rate': rate} for rate in multi_grid])]
    return ResNet_V1_beta(input_shape=input_shape,
                          blocks=blocks,
                          num_classes=num_classes,
                          is_training=is_training,
                          base_only=base_only,
                          global_pool=global_pool,
                          root_block_fn=functools.partial(
                              resnet_v1_utils.root_block_fn),
                          output_stride=output_stride,
                          extract_blocks=extract_blocks,
                          name=name)


def ResNet_V1_101(input_shape=(224, 224, 3),
                  num_classes=None,
                  is_training=None,
                  base_only=False,
                  global_pool=False,
                  min_base_depth=8,
                  depth_multiplier=1,
                  output_stride=None,
                  extract_blocks=None,
                  multi_grid=None,
                  name='resnet_v1_101'):
    if multi_grid is None:
        multi_grid = _DEFAULT_MULTI_GRID
    elif len(multi_grid) != 3:
        raise ValueError("Expect multi_grid to have length 3.")
    depth_func = lambda d: max(int(d * depth_multiplier), min_base_depth)
    blocks = [
        ResNet_V1_beta_block(
            'block1', base_depth=depth_func(64), num_units=3, stride=2),
        ResNet_V1_beta_block(
            'block2', base_depth=depth_func(128), num_units=4, stride=2),
        ResNet_V1_beta_block(
            'block3', base_depth=depth_func(256), num_units=23, stride=2),
        resnet_v1_utils.Block('block4', bottleneck, [
            {'depth': 2048, 'depth_bottleneck': depth_func(512), 'stride': 1,
             'unit_rate': rate} for rate in multi_grid])]
    return ResNet_V1_beta(input_shape=input_shape,
                          blocks=blocks,
                          num_classes=num_classes,
                          base_only=base_only,
                          is_training=is_training,
                          global_pool=global_pool,
                          output_stride=output_stride,
                          extract_blocks=extract_blocks,
                          name=name)


def ResNet_V1_101_beta(input_shape=(224, 224, 3),
                       num_classes=None,
                       is_training=None,
                       base_only=False,
                       global_pool=False,
                       min_base_depth=8,
                       depth_multiplier=1,
                       output_stride=None,
                       extract_blocks=None,
                       multi_grid=None,
                       name='resnet_v1_101'):
    if multi_grid is None:
        multi_grid = _DEFAULT_MULTI_GRID
    elif len(multi_grid) != 3:
        raise ValueError("Expect multi_grid to have length 3.")
    depth_func = lambda d: max(int(d * depth_multiplier), min_base_depth)
    blocks = [
        ResNet_V1_beta_block(
            'block1', base_depth=depth_func(64), num_units=3, stride=2),
        ResNet_V1_beta_block(
            'block2', base_depth=depth_func(128), num_units=4, stride=2),
        ResNet_V1_beta_block(
            'block3', base_depth=depth_func(256), num_units=23, stride=2),
        resnet_v1_utils.Block('block4', bottleneck, [
            {'depth': 2048, 'depth_bottleneck': depth_func(512), 'stride': 1,
             'unit_rate': rate} for rate in multi_grid])]
    return ResNet_V1_beta(input_shape=input_shape,
                          blocks=blocks,
                          num_classes=num_classes,
                          is_training=is_training,
                          base_only=base_only,
                          global_pool=global_pool,
                          root_block_fn=functools.partial(
                              resnet_v1_utils.root_block_fn),
                          output_stride=output_stride,
                          extract_blocks=extract_blocks,
                          name=name)


def ResNet_V1_152(input_shape=(224, 224, 3),
                  num_classes=None,
                  is_training=None,
                  base_only=False,
                  global_pool=False,
                  min_base_depth=8,
                  depth_multiplier=1,
                  output_stride=None,
                  extract_blocks=None,
                  name='resnet_v1_152'):
    depth_func = lambda d: max(int(d * depth_multiplier), min_base_depth)
    blocks = [
        ResNet_V1_beta_block(
            'block1', base_depth=depth_func(64), num_units=3, stride=2),
        ResNet_V1_beta_block(
            'block2', base_depth=depth_func(128), num_units=8, stride=2),
        ResNet_V1_beta_block(
            'block3', base_depth=depth_func(256), num_units=36, stride=2),
        ResNet_V1_beta_block(
            'block4', base_depth=depth_func(512), num_units=3, stride=1)]
    return ResNet_V1_beta(input_shape=input_shape,
                          blocks=blocks,
                          num_classes=num_classes,
                          is_training=is_training,
                          base_only=base_only,
                          global_pool=global_pool,
                          root_block_fn=None,
                          output_stride=output_stride,
                          extract_blocks=extract_blocks,
                          name=name)


def ResNet_V1_200(input_shape=(224, 224, 3),
                  num_classes=None,
                  is_training=None,
                  base_only=False,
                  global_pool=False,
                  min_base_depth=8,
                  depth_multiplier=1,
                  output_stride=None,
                  extract_blocks=None,
                  name='resnet_v1_200'):
    depth_func = lambda d: max(int(d * depth_multiplier), min_base_depth)
    blocks = [
        ResNet_V1_beta_block(
            'block1', base_depth=depth_func(64), num_units=3, stride=2),
        ResNet_V1_beta_block(
            'block2', base_depth=depth_func(128), num_units=24, stride=2),
        ResNet_V1_beta_block(
            'block3', base_depth=depth_func(256), num_units=36, stride=2),
        ResNet_V1_beta_block(
            'block4', base_depth=depth_func(512), num_units=3, stride=1)]

    return ResNet_V1_beta(input_shape=input_shape,
                          blocks=blocks,
                          num_classes=num_classes,
                          is_training=is_training,
                          base_only=base_only,
                          global_pool=global_pool,
                          root_block_fn=None,
                          output_stride=output_stride,
                          extract_blocks=extract_blocks,
                          name=name)


if __name__ == '__main__':
    data = tf.ones((2, 224, 224, 3))
    # with lib.engine.arg_scope(resnet_v1_utils.resnet_arg_scope()):
    #     model = ResNet_V1_101_beta((2, 224, 224, 3), num_classes=1000, is_training=True,
    #                                global_pool=True)
    # print(model(data))
    # for nodes in getattr(model, '_nodes_in_depth'):
    #     for node in nodes:
    #         for weight in node.downstream_layer.weights:
    #             print(weight)
    model = ResNet_V1_18((224, 224, 3), base_only=True,
                         extract_blocks=['pool1', 'block1/unit_2/lite_bottleneck_v1/add',
                                         'block2/unit_2/lite_bottleneck_v1/add',
                                         'block3/unit_2/lite_bottleneck_v1/add',
                                         'block4/unit_2/lite_bottleneck_v1/add'])
    print(model(data))
